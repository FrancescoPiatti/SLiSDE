# v2/src/girsanov/drift_tilt_path.py
"""
Path-based Girsanov tilt controllers.

Where ``drift_tilt.py`` convolves the **Brownian increments** ``dW`` (an
``F_t``-noise process), the controllers in this module convolve the
**Brownian path** ``W_t = ∫_0^t dW_s`` (an ``F_t``-adapted continuous
process; in discrete time this is just ``cumsum(dW)``).  The cumsum is
strictly causal — ``W_k = Σ_{i<k} dW_i`` is ``F_{t_k}``-measurable — so
feeding it into the causal convolution of
``compute_causal_convolution_tilt`` (which already drops the present step
from the kernel sum) preserves adaptability.

Both controllers share the same external call signature as the
``drift_tilt.py`` controllers, so they slot into ``SLiSDEModel`` via the
``tilt_type`` dispatch with no other plumbing change.

Two architectures:

  1. ``PathConvolutionTiltController``
        u^W = K^W ⋆ W_int
        u^Z = K^Z ⋆ Z_eff           (when use_z=True)
        u   = L_mix [u^W, u^Z] + c(t, ξ)

     A near drop-in for ``FFTTiltController`` but driven by the *path*
     instead of the *increments*.  Convolving a path captures levels and
     slow drifts; convolving increments captures local autocorrelation.
     Both are universal causal linear filters for stationary inputs, but
     for non-stationary inputs (which W is — variance grows linearly) the
     two parameterisations have different inductive biases.

  2. ``BilinearPathTiltController``
        u^W   = K^W   ⋆ W_int
        u^Z   = K^Z   ⋆ Z_eff
        u^WZ  = (K^I_W ⋆ W_int) ⊙ (K^I_Z ⋆ Z_eff)        (Hadamard, ``noise_dim``-wise)
        u     = L_mix [u^W, u^Z, u^WZ] + c(t, ξ)

     The interaction term is *second-order* in the joint path, giving the
     controller a poor-man's attention mechanism — the W-path can
     up- or down-weight the Z-driven control depending on the realised
     trajectory.  Still O(T log T) and parameter-light.  Reduces to a
     superset of (1) when ``K^I_W = 0`` or ``K^I_Z = 0`` (warm start
     init).
"""
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.src.girsanov.drift_tilt_utils import compute_causal_convolution_tilt

Array = jnp.ndarray


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _cumsum_path(dW: Array) -> Array:
    """Return the discrete Brownian path ``W_k = Σ_{i<=k} dW_i``.

    Strictly causal because the downstream FFT-conv helper drops the
    present input from the kernel sum (``u_k`` depends only on inputs at
    steps ``0 .. k-1``), so this is exactly the integrated Brownian motion
    sampled at the step grid, and the resulting ``u_k`` depends only on
    ``W_0 .. W_{k-1}``, i.e. on ``dW_0 .. dW_{k-2}`` — strictly causal.
    """
    return jnp.cumsum(dW, axis=1)


def _build_zero_warm_start_mix(
    feature_dim: int,
    noise_dim: int,
    *,
    rngs: nnx.Rngs,
    identity_block: int,
) -> nnx.Linear:
    """``feature_dim → noise_dim`` linear with kernel zero except a tiny
    identity on the first ``identity_block`` columns.

    Mirrors the warm-start convention used by ``drift_tilt.py``: at step 0
    the controller behaves like a small identity on the Brownian sub-block
    plus zero on every other branch, so the tilt is a small perturbation
    of ``u = 0`` at the start of training (ESS ≈ 1).
    """
    layer = nnx.Linear(
        feature_dim, noise_dim,
        rngs=rngs,
        kernel_init=nnx.initializers.zeros,
        bias_init=nnx.initializers.zeros,
    )
    k0 = jnp.zeros(
        (feature_dim, noise_dim), dtype=layer.kernel.value.dtype
    )
    eye_small = 0.01 * jnp.eye(identity_block, noise_dim, dtype=k0.dtype)
    k0 = k0.at[:identity_block, :].set(eye_small)
    layer.kernel.value = k0
    return layer


# -------------------------------------------------------------------------
# 1. Path convolution (linear) — exactly as specified by the user
# -------------------------------------------------------------------------

class PathConvolutionTiltController(nnx.Module):
    """Causal-conv tilt controller driven by the *path*.

    Brownian branch (always present):
        W_int_k = Σ_{i<=k} dW_i^P              # cumsum, F_{t_{k+1}}-measurable
        u^W_k   = (K^W ⋆ W_int)_k              # causal-conv → F_{t_k}-measurable

    State branch (optional, when ``use_z=True``):
        Z_eff_k = R Z^{(L-1)}_k                # optional projection
        u^Z_k   = (K^Z ⋆ Z_eff)_k

    Combine:
        u_k = L_mix [u^W_k, u^Z_k] + c(t_k, ξ)

    The shape contract matches ``FFTTiltController``: input
    ``dW_P : (B, T, noise_dim)``, output ``u : (B, T, noise_dim)``,
    with ``u_k`` ``F_{t_k}``-measurable so that
    ``compute_rn_log_weight(u, dW_P, dt)`` is the correct discrete RN
    log-density of Q wrt P.
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

        # -- Brownian-path branch (always present) --
        # ``W`` grows as O(√t); the existing FFT FFTTiltController uses
        # 0.01·N(0,1) on the *increment* kernel.  Keep the same scale here:
        # at init the conv output ‖K^W ⋆ W_int‖ stays comparable to a
        # one-step ‖dW‖ because the kernel is short and most of the mass
        # comes from the most-recent few entries of ``W``.
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

        # Warm-start mixer: small identity on the W sub-block, zero on the
        # Z sub-block.  Identical convention to drift_tilt.py.
        self.L_mix = _build_zero_warm_start_mix(
            feature_dim=feature_dim,
            noise_dim=noise_dim,
            rngs=rngs,
            identity_block=noise_dim,
        )

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
        # Integrate the increments to get the path; cumsum is strictly
        # causal at the grid the FFT-conv helper subsequently uses.
        W_int = _cumsum_path(dW_P)

        u_W = compute_causal_convolution_tilt(W_int, self.kernel_W[...], None)

        if self.use_z:
            if Z_prev is None:
                raise ValueError(
                    "PathConvolutionTiltController(use_z=True) requires Z_prev"
                )
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
# 2. Bilinear path / state interaction — alternative architecture
# -------------------------------------------------------------------------

class BilinearPathTiltController(nnx.Module):
    """Path-conv tilt controller with a Hadamard interaction term.

    Branches:
        u^W_k   = (K^W   ⋆ W_int)_k
        u^Z_k   = (K^Z   ⋆ Z_eff)_k
        u^WZ_k  = (K^I_W ⋆ W_int)_k ⊙ (K^I_Z ⋆ Z_eff)_k       # element-wise

    Combine:
        u_k = L_mix [u^W_k, u^Z_k, u^WZ_k] + c(t_k, ξ)

    The interaction kernels ``K^I_W`` and ``K^I_Z`` are initialised at
    zero, so at step 0 the controller reduces to the linear
    ``PathConvolutionTiltController`` (above), and the interaction term
    fades in only as the model learns it useful.

    Requires ``use_z=True`` — without a state branch the interaction term
    is meaningless.  Falls back to the linear path controller behaviour
    when both interaction kernels are zero.
    """

    def __init__(
        self,
        noise_dim: int,
        kernel_size: int = 64,
        context_dim: int = 0,
        *,
        use_z: bool = True,
        z_dim: int = 0,
        z_project_dim: int = 0,
        rngs: nnx.Rngs,
    ):
        if not use_z:
            raise ValueError(
                "BilinearPathTiltController requires use_z=True (the "
                "interaction term needs both a W and a Z branch)."
            )
        if z_dim < 1:
            raise ValueError("BilinearPathTiltController requires z_dim >= 1")

        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.use_z = True
        self.z_dim = z_dim
        self.z_project_dim = z_project_dim

        kw, kz, ki_w, ki_z = jr.split(rngs.params(), 4)

        # -- Z projection --
        if z_project_dim > 0:
            self.z_proj = nnx.Linear(
                z_dim, z_project_dim, use_bias=False, rngs=rngs
            )
            z_eff_dim = z_project_dim
        else:
            self.z_proj = None
            z_eff_dim = z_dim

        # -- Linear branches --
        self.kernel_W = nnx.Param(
            0.01 * jr.normal(kw, (kernel_size, noise_dim, noise_dim))
        )
        self.kernel_Z = nnx.Param(
            0.01 * jr.normal(kz, (kernel_size, noise_dim, z_eff_dim))
        )

        # -- Interaction branch (zero-init: starts disabled) --
        # Both factors map into ``noise_dim`` so the Hadamard is well-shaped.
        self.kernel_I_W = nnx.Param(
            jnp.zeros((kernel_size, noise_dim, noise_dim))
        )
        self.kernel_I_Z = nnx.Param(
            jnp.zeros((kernel_size, noise_dim, z_eff_dim))
        )

        # Final mix: 3 branches × noise_dim → noise_dim.  Warm-start with a
        # small identity on the W sub-block (block 0).
        self.L_mix = _build_zero_warm_start_mix(
            feature_dim=3 * noise_dim,
            noise_dim=noise_dim,
            rngs=rngs,
            identity_block=noise_dim,
        )

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
        if Z_prev is None:
            raise ValueError(
                "BilinearPathTiltController requires Z_prev (use_z=True)"
            )
        if Z_prev.shape[1] == dW_P.shape[1] + 1:
            Z_prev = Z_prev[:, :-1, :]

        W_int = _cumsum_path(dW_P)
        Z_eff = self.z_proj(Z_prev) if self.z_proj is not None else Z_prev

        # Linear branches.
        u_W = compute_causal_convolution_tilt(W_int, self.kernel_W[...], None)
        u_Z = compute_causal_convolution_tilt(Z_eff, self.kernel_Z[...], None)

        # Interaction branch: Hadamard product of two causal convs.  Each
        # factor is F_{t_k}-measurable (causal), so the product is also
        # F_{t_k}-measurable.
        u_I_W = compute_causal_convolution_tilt(
            W_int, self.kernel_I_W[...], None
        )
        u_I_Z = compute_causal_convolution_tilt(
            Z_eff, self.kernel_I_Z[...], None
        )
        u_WZ = u_I_W * u_I_Z

        feat = jnp.concatenate([u_W, u_Z, u_WZ], axis=-1)
        u = self.L_mix(feat)

        if self.context_dim > 0 and context is not None:
            c_bias = self.context_proj(context)
            u = u + c_bias[None, None, :]
        return u
