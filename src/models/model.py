# v2/src/models/model.py
"""
v2 SLiSDE model class with config-driven dispatch (Flax NNX).

Stacking modes (unchanged):
  - "residual"           — additive residual connections (no x_prev consumed)
  - "gated_in_flow"      — bilinear-gated additive offset on g
  - "diag_gated_in_flow" — diag-only bilinear scale on F + bilinear residual offset on g

New in v2: optional **last-layer-only Girsanov tilt** (concat+linear,
``drift_tilt2``-style).  Layers 0..L-2 always run under the reference
measure P; only the last layer's Brownian increment is tilted as
``dW^Q = dW^P + u dt`` where the tilt ``u`` is produced by either an
FFT or state-space controller.  The Radon-Nikodym log-weight reflects
the last layer alone.

Forward signature
-----------------
``__call__`` takes ``batch_size`` (P-paths) plus an optional ``batch_size_q``
for a Q-tail.  Returns:

* ``output``                                 — when no Girsanov is active
* ``(output_q, log_rn_q)``                   — single-batch Girsanov (all paths
  under Q, one shared RN log-weight per path)
* ``(output_p, output_q, log_rn_q)``         — dual-batch shared-prefix mode
  (first ``batch_size`` paths under P, last ``batch_size_q`` under Q)

The ``force_no_girsanov`` flag bypasses the tilt at call time, which lets
trainers warm up under P for the first ``start_girsanov_after_epoch``
epochs without rebuilding the model.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.src.config import SLiSDEConfig
from v3.src.noise_gen import NoiseGenerator
from v3.src.layers import (
    SLiSDE,
    ResidualLayer,
    GatedSLiSDELayer,
)
from v3.src.girsanov.drift_tilt import (
    FFTTiltController,
    StateSpaceTiltController,
)
from v3.src.girsanov.drift_tilt_path import (
    PathConvolutionTiltController,
    BilinearPathTiltController,
)
from v3.src.girsanov.drift_tilt_utils import (
    compute_rn_log_weight,
    tilt_brownian_increments,
)

Array = jnp.ndarray


class SLiSDEModel(nnx.Module):
    """v2 Structured Linear Neural SDE model with optional last-layer Girsanov.

    All families are controlled by a single ``SLiSDEConfig``.  When
    ``cfg.use_girsanov`` is False the model returns the output array
    directly (legacy behaviour).  When True it returns a tuple including
    the Radon-Nikodym log-weight for the Q-paths.
    """

    def __init__(self, config: SLiSDEConfig, *, rngs: nnx.Rngs):
        cfg = config
        cfg.validate()
        self.config = cfg
        self.rngs = rngs

        # Noise generator (not an nnx.Module, just a plain object)
        self.noise_gen = NoiseGenerator(mode=cfg.noise_sharing, rho=cfg.noise_rho)

        # Common SDE kwargs
        sde_kwargs = dict(
            dim=cfg.dim,
            noise_dim=cfg.noise_dim,
            matrix_type=cfg.matrix_type,
            block_size=cfg.block_size,
            noise_type=cfg.noise_type,
            w_init_std=cfg.w_init_std,
            approximate_exponential=cfg.approximate_exponential,
            lora_rank=cfg.lora_rank,
        )
        # The merged SLiSDE accepts the time-dependent flag plus embedding
        # hyper-params directly; downstream wrappers forward this dict to
        # their inner ``SLiSDE`` instance.
        base_layer_kwargs = dict(
            time_dependent_vector_fields=cfg.time_dependent_vector_fields,
            time_embed_dim=cfg.time_embed_dim,
            time_omega=cfg.time_omega,
        )
        # Common kwargs for in-flow layers
        in_flow_extra = dict(
            use_causal_conv=cfg.use_causal_conv,
            causal_conv_kernel_size=cfg.causal_conv_kernel_size,
        )

        # First layer (raw SLiSDE).
        self.first_layer = SLiSDE(**sde_kwargs, **base_layer_kwargs, rngs=rngs)

        # Subsequent layers depend on stacking mode.
        # NNX requires modules to be stored as named attributes, not plain lists.
        self._num_extra_layers = cfg.num_layers - 1
        for i in range(1, cfg.num_layers):
            if cfg.stacking_mode == "residual":
                layer = ResidualLayer(
                    **sde_kwargs,
                    norm_type=cfg.norm_type,
                    activation=cfg.activation,
                    base_layer_kwargs=base_layer_kwargs,
                    rngs=rngs,
                )
            elif cfg.stacking_mode == "gated_in_flow":
                # Diag-only F-modulation + bilinear residual g-offset
                # (equivalent to v2's ``diag_gated_in_flow``).
                layer = GatedSLiSDELayer(
                    **sde_kwargs,
                    norm_type=cfg.norm_type,
                    final_scale=cfg.final_scale,
                    use_time_features=cfg.diag_use_time_features,
                    time_omega=cfg.time_omega,
                    base_layer_kwargs=base_layer_kwargs,
                    **in_flow_extra,
                    rngs=rngs,
                )
            else:
                raise ValueError(
                    f"v3 SLiSDEModel: unknown stacking_mode={cfg.stacking_mode!r}"
                )
            setattr(self, f"layer_{i}", layer)

        # Output projection
        self.output_proj = nnx.Linear(cfg.dim, cfg.output_dim, rngs=rngs)

        # Optional learned prior over the initial latent state z0.
        # ``z0_mu`` init at 0 and ``z0_log_sigma`` init at 0 (so σ=1) exactly
        # reproduces the legacy z0 ~ N(0, I) behaviour at step 0 of training.
        if cfg.learn_z0_prior:
            self.z0_mu = nnx.Param(jnp.zeros((cfg.dim,)))
            self.z0_log_sigma = nnx.Param(jnp.zeros((cfg.dim,)))
        else:
            self.z0_mu = None
            self.z0_log_sigma = None

        # ---- Optional last-layer-only Girsanov tilt controller(s) ----
        # When ``cfg.num_girsanov == 1`` we store a single controller as
        # ``self.tilt_controller`` (legacy attribute, unchanged from v2's
        # original layout — ``nnx.PathContains("tilt_controller")`` in
        # dax3's filter still matches it exactly).
        # When ``cfg.num_girsanov > 1`` we store the controllers as
        # ``self.tilt_controller_0``, ``_1`` etc. and ``self.tilt_controller``
        # is left as ``None``.  Filters that need to match every controller
        # in the multi-tilt case must use
        # ``nnx.PathContains("tilt_controller", exact=False)`` (substring).
        self._num_girsanov = int(cfg.num_girsanov) if cfg.use_girsanov else 0
        if cfg.use_girsanov:
            use_z = cfg.girsanov_use_z_dependence
            z_dim = cfg.dim if use_z else 0

            def _make_controller():
                if cfg.tilt_type == "fft":
                    return FFTTiltController(
                        noise_dim=cfg.noise_dim,
                        kernel_size=cfg.kernel_size,
                        context_dim=cfg.context_dim,
                        use_z=use_z,
                        z_dim=z_dim,
                        z_project_dim=cfg.girsanov_z_project_dim,
                        rngs=rngs,
                    )
                if cfg.tilt_type == "state_space":
                    return StateSpaceTiltController(
                        noise_dim=cfg.noise_dim,
                        context_dim=cfg.context_dim,
                        use_z=use_z,
                        z_dim=z_dim,
                        z_project_dim=cfg.girsanov_z_project_dim,
                        n_steps=cfg.n_steps,
                        state_expansion=cfg.girsanov_state_expansion,
                        use_shift=cfg.girsanov_use_shift,
                        rngs=rngs,
                    )
                if cfg.tilt_type == "path_conv":
                    return PathConvolutionTiltController(
                        noise_dim=cfg.noise_dim,
                        kernel_size=cfg.kernel_size,
                        context_dim=cfg.context_dim,
                        use_z=use_z,
                        z_dim=z_dim,
                        z_project_dim=cfg.girsanov_z_project_dim,
                        rngs=rngs,
                    )
                if cfg.tilt_type == "bilinear_path":
                    # Bilinear path controller always needs Z; the config
                    # validator already enforces use_z=True for this tilt_type.
                    return BilinearPathTiltController(
                        noise_dim=cfg.noise_dim,
                        kernel_size=cfg.kernel_size,
                        context_dim=cfg.context_dim,
                        use_z=True,
                        z_dim=cfg.dim,
                        z_project_dim=cfg.girsanov_z_project_dim,
                        rngs=rngs,
                    )
                raise ValueError(
                    f"Unknown tilt_type: {cfg.tilt_type!r} "
                    f"(expected 'fft', 'state_space', 'path_conv' or "
                    f"'bilinear_path')"
                )

            if self._num_girsanov == 1:
                self.tilt_controller = _make_controller()
            else:
                self.tilt_controller = None
                for i in range(self._num_girsanov):
                    setattr(self, f"tilt_controller_{i}", _make_controller())
        else:
            self.tilt_controller = None

    def _tilt_controllers(self) -> list:
        """Return the list of active tilt controllers (length 0 if none)."""
        if self._num_girsanov == 0:
            return []
        if self._num_girsanov == 1:
            return [self.tilt_controller]
        return [getattr(self, f"tilt_controller_{i}")
                for i in range(self._num_girsanov)]

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _run_layer(
        self,
        layer,
        *,
        is_first: bool,
        h_prefix: Optional[Array],
        n_steps: int,
        dt: float,
        batch_size: int,
        z0: Array,
        dW: Array,
    ) -> Array:
        """Run one layer, dispatching on stacking_mode for non-first layers."""
        cfg = self.config
        if is_first:
            return layer(
                n_steps=n_steps, dt=dt, batch_size=batch_size,
                parallel_steps=cfg.parallel_steps,
                z0=z0, dW=dW, return_path=True,
            )
        if cfg.stacking_mode == "residual":
            return h_prefix + layer(
                n_steps=n_steps, dt=dt, batch_size=batch_size,
                parallel_steps=cfg.parallel_steps,
                z0=z0, dW=dW, return_path=True,
            )
        # gated_in_flow
        x_prev = h_prefix[:, :-1, :]
        return layer(
            x_prev=x_prev, n_steps=n_steps, dt=dt,
            batch_size=batch_size, parallel_steps=cfg.parallel_steps,
            z0=z0, dW=dW, return_path=True,
        )

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def __call__(
        self,
        *,
        batch_size: int,
        batch_size_q: int = 0,
        return_path: bool = True,
        z0: Optional[Array] = None,
        y0: Optional[Array] = None,
        context: Optional[Array] = None,
        force_no_girsanov: bool = False,
    ):
        """Run the full v2 model.

        Args:
            batch_size:        Number of P-measure paths.
            batch_size_q:      Number of additional Q-measure paths (only used
                               when Girsanov is active).  When > 0 a
                               *shared-prefix dual batch* is run: layers
                               0..L-2 see ``batch_size + batch_size_q`` paths
                               with untilted dW; the last layer is run with
                               P-dW for the first ``batch_size`` paths and
                               tilted Q-dW for the remaining ``batch_size_q``.
            return_path:       If True return full path, else terminal state.
            z0:                Optional initial latent state.  Sampled if None.
            y0:                Initial observable for y0 enforcement.
            context:           Optional ``(context_dim,)`` xi vector consumed
                               by the tilt controller.  Ignored when Girsanov
                               is inactive or when ``cfg.context_dim == 0``.
            force_no_girsanov: When True, bypass the tilt at call time even
                               if ``cfg.use_girsanov`` is True.  Used by
                               trainers during the warm-up phase.

        Returns:
            * ``output`` array — when Girsanov is inactive.
            * ``(output, log_rn)`` — when Girsanov is active, single
              controller, ``batch_size_q == 0`` (single Q-batch).
            * ``(output_p, output_q, log_rn_q)`` — single-controller
              dual-batch shared-prefix.
            * ``(output_p, [output_q_i], [log_rn_q_i])`` — multi-controller
              dual-batch (``num_girsanov > 1``).  Each controller has its
              own Q-batch of size ``batch_size_q``.
        """
        cfg = self.config
        dt = cfg.T / cfg.n_steps

        controllers = self._tilt_controllers()
        N_g = len(controllers)
        girsanov_active = (
            cfg.use_girsanov and (not force_no_girsanov) and N_g > 0
        )
        dual = girsanov_active and (batch_size_q > 0)
        if girsanov_active and N_g > 1 and not dual:
            raise ValueError(
                "num_girsanov > 1 requires dual-batch mode "
                "(batch_size_q > 0); single-batch (all-Q) is single-tilt only."
            )
        n_p = batch_size
        n_q = batch_size_q if dual else 0
        # Layout of the shared-prefix batch:
        #   indices [0, n_p)               : P paths
        #   indices [n_p + i*n_q, n_p + (i+1)*n_q) : Q paths for controller i
        # Single-batch Girsanov (single-tilt only): all paths under controller 0.
        if girsanov_active and not dual and N_g == 1:
            # Single-batch: ``n_p`` paths all driven by the single tilt.
            B = n_p
        else:
            B = n_p + N_g * n_q

        # ---- Initial state ----
        if z0 is None:
            key0 = self.rngs.init()
            eps = jr.normal(key0, (B, cfg.dim))
            if self.z0_mu is not None:
                z0 = self.z0_mu[None, :] + jnp.exp(self.z0_log_sigma)[None, :] * eps
            else:
                z0 = eps

        # ---- Sample dW^P ----
        noise_key = self.rngs.noise()
        dW_all = self.noise_gen.sample(
            noise_key,
            n_steps=cfg.n_steps,
            noise_dim=cfg.noise_dim,
            T=cfg.T,
            num_blocks=cfg.num_layers,
            batch_size=B,
            augment_with_time=cfg.augment_brownian_with_time,
        )

        def _get_dW(layer_idx: int) -> Array:
            if cfg.num_layers == 1:
                return dW_all
            return dW_all[:, layer_idx]

        L = cfg.num_layers
        last_idx = L - 1

        # ---- Run prefix layers 0..L-2 (under P) ----
        if L == 1:
            h_prefix = None
        else:
            h_prefix = self._run_layer(
                self.first_layer, is_first=True, h_prefix=None,
                n_steps=cfg.n_steps, dt=dt, batch_size=B,
                z0=z0, dW=_get_dW(0),
            )
            for l_idx in range(1, L - 1):
                layer = getattr(self, f"layer_{l_idx}")
                h_prefix = self._run_layer(
                    layer, is_first=False, h_prefix=h_prefix,
                    n_steps=cfg.n_steps, dt=dt, batch_size=B,
                    z0=z0, dW=_get_dW(l_idx),
                )

        # ---- Compute tilt(s) for last layer ----
        dW_last_P = _get_dW(last_idx)  # (B, n_steps, noise_dim)

        if girsanov_active:
            # Correlated last-layer Brownian motion (see
            # ``v2/latex/girsanov_correlated_lastlayer.tex``).
            #
            # Per controller i:
            #   1. Sample a fresh orthogonal increment ``eps_tilde_i`` (the
            #      Q-Brownian on the orthogonal component).
            #   2. Compute the controller drift ``v_i = controller(eps_tilde_i)``
            #      directly on this orthogonal noise.  ``sqrt(1 - rho^2)`` is
            #      absorbed into v.
            #   3. Reconstruct the P-orthogonal increment
            #      ``eps_P_i = eps_tilde_i + v_i dt``.
            #   4. Radon-Nikodym log-weight (clean Novikov, no rho factor):
            #         log_rn_i = sum_k v_i[k] . eps_P_i[k] - 0.5 |v_i[k]|^2 dt
            #   5. Build the last-layer SDE driver:
            #         tilde_dW_i = rho * dW_shared + sqrt(1 - rho^2) * eps_tilde_i
            #      (For augmented Brownian, override the time channel to dt.)
            rho = jnp.asarray(cfg.girsanov_last_layer_rho, dtype=dW_last_P.dtype)
            sqrt_one_minus_rho2 = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))
            augment_time = cfg.augment_brownian_with_time

            log_rn_list: list = []
            tilde_dW_q_list: list = []
            for i, controller in enumerate(controllers):
                if dual:
                    start = n_p + i * n_q
                    stop = n_p + (i + 1) * n_q
                    dW_shared_i = dW_last_P[start:stop]
                    Z_slice = h_prefix[start:stop] if h_prefix is not None else None
                    n_q_i = n_q
                else:
                    dW_shared_i = dW_last_P
                    Z_slice = h_prefix
                    n_q_i = n_p

                # 1. Fresh orthogonal increment eps_tilde_i (one fresh draw per
                #    controller — distinct controllers must sample independent
                #    orthogonal Brownians).
                eps_key = self.rngs.noise()
                eps_tilde_i = self.noise_gen.sample_lastlayer_orth(
                    eps_key,
                    n_steps=cfg.n_steps,
                    noise_dim=cfg.noise_dim,
                    T=cfg.T,
                    batch_size=n_q_i,
                    augment_with_time=augment_time,
                )

                if cfg.girsanov_use_z_dependence:
                    if Z_slice is None:
                        raise ValueError(
                            "girsanov_use_z_dependence=True requires num_layers >= 2"
                        )
                    # Sever the SLiSDE backbone from SNIS-noise gradients that
                    # would otherwise flow through the controller's Z-branch
                    # via the shared prefix.
                    Z_for_tilt = jax.lax.stop_gradient(Z_slice)
                else:
                    Z_for_tilt = None

                # 2. Controller drift on the orthogonal noise.
                v_i = controller(
                    dW_P=eps_tilde_i,
                    dt=dt,
                    n_steps=cfg.n_steps,
                    context=context,
                    parallel_steps=cfg.parallel_steps,
                    Z_prev=Z_for_tilt,
                )

                # When the noise tensor carries a deterministic time channel,
                # zero out v on that channel — there is nothing to tilt there
                # (the time channel is not a Brownian channel under either
                # measure), and including it would inflate the RN penalty and
                # add a spurious deterministic shift to the time feature.
                if augment_time:
                    mask = jnp.concatenate(
                        [
                            jnp.ones((cfg.noise_dim - 1,), dtype=v_i.dtype),
                            jnp.zeros((1,), dtype=v_i.dtype),
                        ],
                        axis=0,
                    )
                    v_i = v_i * mask[None, None, :]

                # 3. P-orthogonal increment.
                eps_P_i = eps_tilde_i + v_i * dt

                # 4. Radon-Nikodym log-weight (clean Novikov form, no rho).
                log_rn_list.append(compute_rn_log_weight(v_i, eps_P_i, dt))

                # 5. SDE driver for the last layer.
                tilde_dW_i = rho * dW_shared_i + sqrt_one_minus_rho2 * eps_tilde_i
                if augment_time:
                    # Restore the deterministic time channel = dt.
                    time_shape = tilde_dW_i.shape[:-1] + (1,)
                    time_channel = jnp.full(time_shape, dt, dtype=tilde_dW_i.dtype)
                    tilde_dW_i = jnp.concatenate(
                        [tilde_dW_i[..., :-1], time_channel], axis=-1
                    )
                tilde_dW_q_list.append(tilde_dW_i)

            if dual:
                dW_last_eff = jnp.concatenate(
                    [dW_last_P[:n_p]] + tilde_dW_q_list, axis=0
                )
            else:
                dW_last_eff = tilde_dW_q_list[0]
        else:
            dW_last_eff = dW_last_P
            log_rn_list = []
            tilde_dW_q_list = []

        # ---- Run last layer ----
        if L == 1:
            h = self._run_layer(
                self.first_layer, is_first=True, h_prefix=None,
                n_steps=cfg.n_steps, dt=dt, batch_size=B,
                z0=z0, dW=dW_last_eff,
            )
        else:
            last_layer = getattr(self, f"layer_{last_idx}")
            h = self._run_layer(
                last_layer, is_first=False, h_prefix=h_prefix,
                n_steps=cfg.n_steps, dt=dt, batch_size=B,
                z0=z0, dW=dW_last_eff,
            )

        # ---- Output projection ----
        output = self.output_proj(h)

        # y0 enforcement
        if y0 is not None:
            shift = y0[None, None, :] - output[:, 0:1, :]
            output = output + shift

        if not return_path:
            output = output[:, -1, :]

        # ---- Return ----
        if not girsanov_active:
            return output
        if dual:
            if N_g == 1:
                # Backward-compatible single-tilt return shape.
                return output[:n_p], output[n_p:], log_rn_list[0]
            # Multi-tilt: split the Q-section back into per-controller chunks.
            outputs_q = [
                output[n_p + i * n_q : n_p + (i + 1) * n_q]
                for i in range(N_g)
            ]
            return output[:n_p], outputs_q, log_rn_list
        # Single-batch (single-tilt only — multi-tilt is rejected above).
        return output, log_rn_list[0]
