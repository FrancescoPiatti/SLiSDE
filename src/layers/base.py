"""Base structured-linear SDE layer (v3).

The single :class:`SLiSDE` class covers both the time-homogeneous case and
the time-dependent variant that v2 split out as ``SLiSDETime3``. Toggle
``time_dependent_vector_fields=True`` to make the structured coefficients
``A, b, C^j, D`` functions of time via a small zero-initialised decoder
on the deterministic features ``τ(t) = (1, t, t², sin ωt, cos ωt)``.

The decoder is *structure-aware*: with ``matrix_type="diag"`` we decode
only diagonal entries, with ``"blockdiag"`` only block entries, with
``"dense"`` full matrices. Zero-init means the layer reduces exactly to
the time-homogeneous SDE at step 0 of training.
"""
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.src.structured_matrices import (
    create_drift_params,
    create_diffusion_params,
    create_dense_diffusion_params,
    create_lora_diffusion_params,
    create_bias_param,
)
from v3.src.flows import (
    build_affine_flows_diag,
    build_affine_flows_blockdiag,
    build_affine_flows_dense,
    inject_offset_diag,
    inject_offset_blockdiag,
    inject_offset_dense,
    affine_step_diag,
    affine_step_blockdiag,
    affine_step_dense,
    parallel_affine_step_diag,
    parallel_affine_step_blockdiag,
    parallel_affine_step_dense,
)
from v3.src.utils import time_feature_vector, TIME_FEATURE_DIM

Array = jnp.ndarray
MatrixType = Literal["diag", "blockdiag", "dense"]
NoiseType = Literal["multiplicative", "additive", "general"]


class SLiSDE(nnx.Module):
    """Structured linear SDE layer.

    Continuous form::

        dZ = (A_t Z + b_t) dt + Σ_j C^j_t Z dW^j + D_t dW

    Discretised to the affine recurrence ``Z_{k+1} = F_k Z_k + g_k``, which
    is solved with ``jax.lax.scan`` (sequential) or a chunked associative
    scan (parallel).

    Setting ``time_dependent_vector_fields=True`` activates a small
    coefficient decoder ``coeff_t = coeff + W · tanh(U τ(t) + c)`` on every
    structured coefficient (drift, bias, multiplicative diffusion, additive
    diffusion). All decoders are zero-initialised, so the model starts from
    the time-homogeneous structured SDE and learns time variation only when
    useful.
    """

    def __init__(
        self,
        dim: int,
        noise_dim: int,
        matrix_type: MatrixType = "blockdiag",
        block_size: int = 8,
        noise_type: NoiseType = "multiplicative",
        w_init_std: float = 0.25,
        approximate_exponential: bool = True,
        lora_rank: int = 0,
        time_dependent_vector_fields: bool = False,
        time_embed_dim: int = 8,
        time_omega: float = 6.283185307179586,  # 2π
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.noise_dim = noise_dim
        self.matrix_type = matrix_type
        self.block_size = block_size
        self.noise_type = noise_type
        self.w_init_std = w_init_std
        self.approximate_exponential = approximate_exponential
        self.lora_rank = lora_rank
        self.time_dependent_vector_fields = bool(time_dependent_vector_fields)
        self.time_embed_dim = int(time_embed_dim)
        self.time_omega = float(time_omega)

        # -- Structured drift / diffusion parameters -----------------------
        k_A, k_b, k_C, k_D, k_t = jr.split(rngs.params(), 5)
        self.A = nnx.Param(create_drift_params(k_A, matrix_type, dim, block_size, w_init_std))
        self.b = nnx.Param(create_bias_param(k_b, dim))

        if noise_type in ("multiplicative", "general"):
            self.C = nnx.Param(create_diffusion_params(
                k_C, matrix_type, dim, noise_dim, block_size, w_init_std,
            ))
        else:
            self.C = None

        if noise_type in ("additive", "general"):
            if lora_rank > 0:
                D_A, D_B = create_lora_diffusion_params(k_D, dim, noise_dim, lora_rank, w_init_std)
                self.D_A = nnx.Param(D_A)
                self.D_B = nnx.Param(D_B)
                self.dense_D = None
            else:
                self.D_A = None
                self.D_B = None
                self.dense_D = nnx.Param(create_dense_diffusion_params(
                    k_D, dim, noise_dim, w_init_std,
                ))
        else:
            self.D_A = None
            self.D_B = None
            self.dense_D = None

        # -- Optional time-dependent coefficient decoders ------------------
        if self.time_dependent_vector_fields:
            self._init_time_decoders(k_t)
        else:
            self.time_U = None
            self.time_c = None
            self.A_decoder = None
            self.b_decoder = None
            self.C_decoder = None
            self.D_decoder = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_time_decoders(self, key):
        """Zero-initialise the structure-aware coefficient decoders."""
        H = self.time_embed_dim
        k_U, _ = jr.split(key, 2)
        init_std = 1.0 / jnp.sqrt(jnp.asarray(TIME_FEATURE_DIM, dtype=jnp.float32))
        # τ(t) -> h(t) embedding; only this matrix has non-zero init so the
        # decoded corrections still start at zero (the per-coefficient ``W``
        # matrices below are zero-init).
        self.time_U = nnx.Param(init_std * jr.normal(k_U, (TIME_FEATURE_DIM, H)))
        self.time_c = nnx.Param(jnp.zeros((H,)))

        # Structure-aware per-coefficient decoders.
        if self.matrix_type == "diag":
            self.A_decoder = nnx.Param(jnp.zeros((self.dim, H)))
        elif self.matrix_type == "blockdiag":
            nb = self.dim // self.block_size
            self.A_decoder = nnx.Param(
                jnp.zeros((nb, self.block_size, self.block_size, H))
            )
        else:  # dense
            self.A_decoder = nnx.Param(jnp.zeros((self.dim, self.dim, H)))

        self.b_decoder = nnx.Param(jnp.zeros((self.dim, H)))

        if self.C is not None:
            if self.matrix_type == "diag":
                self.C_decoder = nnx.Param(jnp.zeros((self.noise_dim, self.dim, H)))
            elif self.matrix_type == "blockdiag":
                nb = self.dim // self.block_size
                self.C_decoder = nnx.Param(jnp.zeros(
                    (self.noise_dim, nb, self.block_size, self.block_size, H)
                ))
            else:
                self.C_decoder = nnx.Param(
                    jnp.zeros((self.noise_dim, self.dim, self.dim, H))
                )
        else:
            self.C_decoder = None

        if self.noise_type in ("additive", "general"):
            # Per-noise-channel amplitude time gate on D — shape (H, m).
            self.D_decoder = nnx.Param(jnp.zeros((H, self.noise_dim)))
        else:
            self.D_decoder = None

    # ------------------------------------------------------------------
    # Forward-pass helpers
    # ------------------------------------------------------------------

    def _get_dense_D(self):
        """Return the effective dense ``D`` matrix (collapses LoRA factors)."""
        if self.dense_D is not None:
            return self.dense_D[...]
        if self.D_A is not None:
            return self.D_A[...] @ self.D_B[...]
        return None

    def _compute_h(self, n_steps: int, dtype) -> Array:
        """Return the time embedding ``h(t)`` of shape ``(n_steps, H)``.

        Only called when ``time_dependent_vector_fields=True``.
        """
        tau = time_feature_vector(n_steps, self.time_omega, dtype=dtype)
        return jnp.tanh(tau @ self.time_U[...] + self.time_c[...])

    def _build_flows(self, dt, dW):
        """Construct the ``(F, g)`` flows for this layer.

        Dispatches to the matrix-type-specific flow builder, passing decoder
        arrays and the time embedding when the layer is time-dependent.
        """
        dense_D = self._get_dense_D()
        C = self.C[...] if self.C is not None else None

        if self.time_dependent_vector_fields:
            h_t = self._compute_h(dW.shape[1], dW.dtype)
            decoders = dict(
                h_t=h_t,
                A_decoder=self.A_decoder[...],
                b_decoder=self.b_decoder[...],
                C_decoder=self.C_decoder[...] if self.C_decoder is not None else None,
                D_decoder=self.D_decoder[...] if self.D_decoder is not None else None,
            )
        else:
            decoders = {}

        kwargs = dict(
            A=self.A[...],
            b=self.b[...],
            dt=dt,
            dW=dW,
            noise_type=self.noise_type,
            approx_exp=self.approximate_exponential,
            C=C,
            dense_D=dense_D,
            **decoders,
        )
        if self.matrix_type == "diag":
            return build_affine_flows_diag(**kwargs)
        if self.matrix_type == "blockdiag":
            return build_affine_flows_blockdiag(**kwargs)
        return build_affine_flows_dense(**kwargs)

    def _get_step_fns(self):
        """Return (sequential_step, parallel_step) for the current matrix type."""
        if self.matrix_type == "diag":
            return affine_step_diag, parallel_affine_step_diag
        if self.matrix_type == "blockdiag":
            return affine_step_blockdiag, parallel_affine_step_blockdiag
        return affine_step_dense, parallel_affine_step_dense

    def _inject_offset(self, flows, offset):
        """Add a gate-produced offset to the affine intercept ``g``."""
        if self.matrix_type == "diag":
            return inject_offset_diag(flows, offset)
        if self.matrix_type == "blockdiag":
            return inject_offset_blockdiag(flows, offset)
        return inject_offset_dense(flows, offset)

    def _run_affine_scan(self, z0, flows, parallel_steps):
        """Run the affine scan (sequential or chunked-parallel) over one path."""
        n_steps = jax.tree_util.tree_leaves(flows)[0].shape[0]
        step_fn, pstep_fn = self._get_step_fns()

        if parallel_steps == 1:
            _, zs = jax.lax.scan(step_fn, z0, flows)
            return jnp.concatenate([z0[None, :], zs], axis=0)

        # Chunked parallel scan: split off any remainder for a tail sequential pass.
        remainder = n_steps % parallel_steps
        if remainder == 0:
            core, rem = flows, None
        else:
            core = jax.tree_util.tree_map(lambda x: x[:-remainder], flows)
            rem = jax.tree_util.tree_map(lambda x: x[-remainder:], flows)

        core_chunks = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, parallel_steps, *x.shape[1:]), core,
        )

        def outer_step(z, chunk):
            return pstep_fn(z, chunk)

        _, zs_chunks = jax.lax.scan(outer_step, z0, core_chunks)
        zs_core = zs_chunks.reshape(-1, self.dim)
        zs_all = jnp.concatenate([z0[None, :], zs_core], axis=0)

        if rem is not None:
            z_last = zs_all[-1]
            _, zs_rem = jax.lax.scan(step_fn, z_last, rem)
            zs_all = jnp.concatenate([zs_all, zs_rem], axis=0)

        return zs_all

    # ------------------------------------------------------------------
    # Forward call
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        n_steps,
        dt=1.0,
        batch_size=1,
        parallel_steps=1,
        z0=None,
        dW=None,
        return_path=True,
        external_offset=None,
    ):
        """Run the SDE layer.

        Args:
            n_steps: Number of time steps (kept for API parity; ``dW`` already
                encodes the time dimension).
            dt: Time step size.
            batch_size: Number of paths (kept for API parity).
            parallel_steps: Chunk size for parallel scan (1 = sequential).
            z0: Initial state ``(B, d)``. Must be provided.
            dW: Brownian increments ``(B, T, m)``. Must be provided.
            return_path: If True return the full path ``(B, T+1, d)``;
                otherwise the terminal state ``(B, d)``.
            external_offset: Optional gate-produced ``(B, T, d)`` to add to
                the affine intercept.
        """
        dt = jnp.asarray(dt)
        flows = self._build_flows(dt, dW)
        if external_offset is not None:
            flows = self._inject_offset(flows, external_offset)

        F, g = flows

        def run_one(z0_b, F_b, g_b):
            return self._run_affine_scan(z0_b, (F_b, g_b), parallel_steps)

        paths = jax.vmap(run_one)(z0, F, g)
        return paths if return_path else paths[:, -1, :]
