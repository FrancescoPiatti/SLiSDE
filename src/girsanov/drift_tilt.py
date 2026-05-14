# v2/src/girsanov/drift_tilt.py
"""
Last-layer-only Girsanov tilt controllers (concat+linear variant).

v2 keeps a single Girsanov family — the ``drift_tilt2.py`` design from v1 —
because (a) it strictly subsumes the legacy Brownian-only controllers via
the warm-started ``L_mix`` initialisation, and (b) its optional
``Z^{(L-1)}`` branch lets the tilt depend on the prefix state without
needing an extra untilted batch.

Both controllers compute

    u^W_k = ConvOrSSM_W(dW_P)
    u^Z_k = ConvOrSSM_Z(Z_eff)         (only when use_z=True)
    u_k   = L_mix [u^W_k, u^Z_k] + c(t_k, xi)

where ``L_mix`` is initialised to a small identity on the Brownian sub-
block (so the tilt contributes from step 0 of training) and zero on the
``Z`` sub-block (so the Z branch fades in gradually, avoiding an early
ESS crash).

Where the v1 implementation fixed ``state_dim_W = noise_dim`` for the
state-space variant we keep the same convention here, so behaviour
matches drift_tilt2.py exactly.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.src.girsanov.drift_tilt_utils import (
    compute_causal_convolution_tilt,
    build_controller_affine_flows,
    run_controller_parallel_scan,
)

Array = jnp.ndarray


# -------------------------------------------------------------------------
# FFT variant
# -------------------------------------------------------------------------

class FFTTiltController(nnx.Module):
    """FFT-based causal-convolution tilt controller (concat+linear).

    Brownian branch (always present):
        u^W_k = (K^W * dW^P)_k

    State branch (optional, when ``use_z=True``):
        Z_eff_k = R Z^{(L-1)}_k        (R: identity if z_project_dim==0)
        u^Z_k   = (K^Z * Z_eff)_k

    Combine:
        u_k = L_mix [u^W_k, u^Z_k] + c(t_k, xi)

    When ``use_z=False`` this reduces to ``u_k = L_mix u^W_k + c(t_k, xi)``,
    which is the legacy Brownian-only FFT tilt plus a final
    ``noise_dim x noise_dim`` dense mix.
    """

    def __init__(
        self,
        noise_dim: int,
        kernel_size: int = 64,
        context_dim: int = 0,
        *,
        use_z: bool = False,
        z_dim: int = 0,
        z_project_dim: int = 0,
        rngs: nnx.Rngs,
    ):
        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.use_z = use_z
        self.z_dim = z_dim
        self.z_project_dim = z_project_dim

        kw, kz, _ = jr.split(rngs.params(), 3)

        # -- Brownian branch --
        self.kernel_W = nnx.Param(
            0.01 * jr.normal(kw, (kernel_size, noise_dim, noise_dim))
        )

        # -- State branch (optional) --
        if use_z:
            if z_dim < 1:
                raise ValueError("use_z=True requires z_dim >= 1")
            if z_project_dim > 0:
                self.z_proj = nnx.Linear(
                    z_dim, z_project_dim, use_bias=False, rngs=rngs
                )
                z_eff_dim = z_project_dim
            else:
                self.z_proj = None
                z_eff_dim = z_dim
            self.kernel_Z = nnx.Param(
                0.01 * jr.normal(kz, (kernel_size, noise_dim, z_eff_dim))
            )
            feature_dim = 2 * noise_dim
        else:
            self.z_proj = None
            self.kernel_Z = None
            feature_dim = noise_dim

        # -- Final dense mixing layer --
        # Warm start: small identity on Brownian sub-block, zero on Z sub-block.
        self.L_mix = nnx.Linear(
            feature_dim, noise_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )
        k0 = jnp.zeros(
            (feature_dim, noise_dim), dtype=self.L_mix.kernel.value.dtype
        )
        eye_small = 0.01 * jnp.eye(noise_dim, dtype=k0.dtype)
        k0 = k0.at[:noise_dim, :noise_dim].set(eye_small)
        self.L_mix.kernel.value = k0

        if context_dim > 0:
            self.context_proj = nnx.Linear(context_dim, noise_dim, rngs=rngs)

    def __call__(
        self,
        dW_P: Array,
        dt: Array,
        n_steps: int,
        context: Optional[Array] = None,
        parallel_steps: int = 1,
        Z_prev: Optional[Array] = None,
    ) -> Array:
        # Brownian branch
        u_W = compute_causal_convolution_tilt(dW_P, self.kernel_W[...], None)

        # State branch
        if self.use_z:
            if Z_prev is None:
                raise ValueError("FFTTiltController(use_z=True) requires Z_prev")
            if Z_prev.shape[1] == dW_P.shape[1] + 1:
                Z_prev = Z_prev[:, :-1, :]
            Z_eff = self.z_proj(Z_prev) if self.z_proj is not None else Z_prev
            u_Z = compute_causal_convolution_tilt(
                Z_eff, self.kernel_Z[...], None
            )
            feat = jnp.concatenate([u_W, u_Z], axis=-1)
        else:
            feat = u_W

        u = self.L_mix(feat)

        if self.context_dim > 0 and context is not None:
            c_bias = self.context_proj(context)
            u = u + c_bias[None, None, :]
        return u


# -------------------------------------------------------------------------
# State-space variant
# -------------------------------------------------------------------------

class StateSpaceTiltController(nnx.Module):
    """Diagonal state-space tilt controller (concat+linear).

    Brownian branch (always present):
        s^W_{k+1} = diag(rho_W) s^W_k + U_W dW_k^P

    State branch (optional, when ``use_z=True``):
        Z_eff_k   = R Z^{(L-1)}_k    (optional projection)
        s^Z_{k+1} = diag(rho_Z) s^Z_k + U_Z Z_eff_k

    Combine:
        u_k = L_mix [s^W_k, s^Z_k] + c(t_k, xi)

    When ``use_z=False`` reduces to ``u_k = L_mix s^W_k + c(t_k, xi)``.
    """

    def __init__(
        self,
        noise_dim: int,
        context_dim: int = 0,
        *,
        use_z: bool = False,
        z_dim: int = 0,
        z_project_dim: int = 0,
        n_steps: Optional[int] = None,
        state_expansion: int = 1,
        use_shift: bool = False,
        rngs: nnx.Rngs,
    ):
        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.use_z = use_z
        self.z_dim = z_dim
        self.z_project_dim = z_project_dim
        self.use_shift = use_shift

        # Recurrent-state dim — ``state_expansion`` lets the controller carry
        # more channels than the noise itself, which gives it strictly more
        # capacity to summarise the past dW history without changing the
        # output dim.  The extra channels feed into ``L_mix`` whose kernel
        # rows are zero-initialised, so init behaviour is unchanged.
        if state_expansion < 1:
            raise ValueError("state_expansion must be >= 1")
        self.state_dim_W = state_expansion * noise_dim

        _, kw_U, _, kz_U = jr.split(rngs.params(), 4)

        # -- Brownian branch --
        self.log_rho_W = nnx.Param(jnp.zeros((self.state_dim_W,)))
        self.U_W = nnx.Param(
            0.01 * jr.normal(kw_U, (self.state_dim_W, noise_dim))
        )

        # -- State branch (optional) --
        if use_z:
            if z_dim < 1:
                raise ValueError("use_z=True requires z_dim >= 1")
            if z_project_dim > 0:
                self.z_proj = nnx.Linear(
                    z_dim, z_project_dim, use_bias=False, rngs=rngs
                )
                z_eff_dim = z_project_dim
            else:
                self.z_proj = None
                z_eff_dim = z_dim
            self.state_dim_Z = z_eff_dim
            self.log_rho_Z = nnx.Param(jnp.zeros((self.state_dim_Z,)))
            self.U_Z = nnx.Param(
                0.01 * jr.normal(kz_U, (self.state_dim_Z, z_eff_dim))
            )
            feature_dim = self.state_dim_W + self.state_dim_Z
        else:
            self.z_proj = None
            self.state_dim_Z = 0
            self.log_rho_Z = None
            self.U_Z = None
            feature_dim = self.state_dim_W

        # -- Final dense mixing layer --
        self.L_mix = nnx.Linear(
            feature_dim, noise_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )
        k0 = jnp.zeros(
            (feature_dim, noise_dim), dtype=self.L_mix.kernel.value.dtype
        )
        # Init: only the *first* ``noise_dim`` rows of the W-block carry the
        # 0.01·I warm-start; any extra rows from ``state_expansion > 1`` and
        # the Z-block remain zero, so init behaviour is independent of
        # ``state_expansion`` and ``use_z``.
        eye_small = 0.01 * jnp.eye(noise_dim, dtype=k0.dtype)
        k0 = k0.at[:noise_dim, :].set(eye_small)
        self.L_mix.kernel.value = k0

        if context_dim > 0:
            self.context_proj = nnx.Linear(context_dim, noise_dim, rngs=rngs)

        # Per-time-step learnable bias on ``u_k``.  Initialised to zero so
        # the controller starts unchanged; gradient descent through the
        # SNIS barrier loss can then learn a deterministic time-varying
        # tilt directly, without having to express it through the recurrent
        # W-state.  Particularly useful when the optimal tilt is close to
        # a fixed drift (e.g. push paths toward a barrier).
        if use_shift:
            if n_steps is None or n_steps < 1:
                raise ValueError(
                    "use_shift=True requires n_steps to be a positive int"
                )
            self.c_shift = nnx.Param(jnp.zeros((n_steps, noise_dim)))
        else:
            self.c_shift = None

    @staticmethod
    def _run_controller(
        rho: Array, U: Array, driver: Array, parallel_steps: int
    ) -> Array:
        state_dim = U.shape[0]
        F_ctrl, g_ctrl = build_controller_affine_flows(rho, U, driver)
        s0 = jnp.zeros(state_dim)

        def run_one(F_b, g_b):
            return run_controller_parallel_scan(s0, F_b, g_b, parallel_steps)

        s_all = jax.vmap(run_one)(F_ctrl, g_ctrl)
        # Drop trailing s_T so that u_k depends only on dW_0..dW_{k-1}.
        return s_all[:, :-1, :]

    def __call__(
        self,
        dW_P: Array,
        dt: Array,
        n_steps: int,
        context: Optional[Array] = None,
        parallel_steps: int = 1,
        Z_prev: Optional[Array] = None,
    ) -> Array:
        rho_W = jnp.exp(self.log_rho_W[...])
        s_W = self._run_controller(rho_W, self.U_W[...], dW_P, parallel_steps)

        if self.use_z:
            if Z_prev is None:
                raise ValueError(
                    "StateSpaceTiltController(use_z=True) requires Z_prev"
                )
            if Z_prev.shape[1] == dW_P.shape[1] + 1:
                Z_prev = Z_prev[:, :-1, :]
            Z_eff = self.z_proj(Z_prev) if self.z_proj is not None else Z_prev
            rho_Z = jnp.exp(self.log_rho_Z[...])
            s_Z = self._run_controller(
                rho_Z, self.U_Z[...], Z_eff, parallel_steps
            )
            feat = jnp.concatenate([s_W, s_Z], axis=-1)
        else:
            feat = s_W

        u = self.L_mix(feat)

        if self.use_shift and self.c_shift is not None:
            T = u.shape[1]
            shift = self.c_shift[...]
            # Defensive slice: use leading T entries if the eval grid is
            # shorter; pad with zeros at the end if longer.
            if shift.shape[0] >= T:
                shift = shift[:T]
            else:
                pad = jnp.zeros(
                    (T - shift.shape[0], shift.shape[1]), dtype=shift.dtype
                )
                shift = jnp.concatenate([shift, pad], axis=0)
            u = u + shift[None, :, :]

        if self.context_dim > 0 and context is not None:
            c_bias = self.context_proj(context)
            u = u + c_bias[None, None, :]
        return u
