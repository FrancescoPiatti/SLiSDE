"""Gated in-flow SLiSDE stacking layer (v3).

This is the single in-flow stacking option in v3 — the v2 split between
``gated_in_flow`` (full-row F-scale, bilinear gate) and
``diag_gated_in_flow`` (diag-only F-scale, bilinear gate + bilinear
residual offset on g) is collapsed into the second, more conservative
variant, which v3 simply calls ``gated_in_flow``.

For each time step ``k`` the layer:

1. (optionally) passes the previous-layer path ``x_prev`` through a
   small causal 1-D convolution wrapped in a residual skip;
2. normalises it via LayerNorm or RMSNorm;
3. (optionally) concatenates deterministic time features
   ``(t, t², sin ωt, cos ωt)`` to keep them out of the norm;
4. builds two bilinear ``σ ⊙ tanh`` / ``σ ⊙ id`` gates that produce a
   bounded diagonal scale ``α_k`` (cap ``ε = final_scale``) and an
   affine offset ``o_k``;
5. modifies the inner SDE's affine pair as
        F_k[..., j, j] ← α_k[..., j] · F_k[..., j, j]    (diag-only scale)
        g_k            ← g_k + o_k                        (residual offset)
   while leaving off-diagonal entries of ``F_k`` untouched.

The structure (diag / blockdiag / dense) is preserved because we only
modulate entries that already live on the (block) diagonal. Gate kernels
are initialised at ``normal(0.01)`` so ``α_k ≈ 1`` and ``o_k ≈ 0`` at
step 0 — the stacked model starts close to an unmodulated structured
SDE and learns cross-layer coupling gradually.
"""
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from .base import SLiSDE
from v3.src.utils import time_feature_vector

Array = jnp.ndarray
MatrixType = Literal["diag", "blockdiag", "dense"]
NoiseType = Literal["multiplicative", "additive", "general"]
NormType = Literal["layer", "rms"]

# Number of deterministic time channels appended to the gate input when
# ``use_time_features=True``: (t, t², sin ωt, cos ωt). The constant column
# is dropped because the gate bias already supplies a constant channel.
TIME_AUG_DIM = 4


class CausalConv1D(nnx.Module):
    """1-D causal convolution along the time axis with a residual skip.

    Inputs / outputs are ``(B, T, dim)``. The input is left-padded by
    ``kernel_size − 1`` zeros to enforce causality without shrinking ``T``.
    Kernel init at ``normal(0.01)`` plus the residual skip ⇒ the layer
    starts at ≈ identity, matching the no-conv layer at step 0.
    """

    def __init__(self, dim: int, kernel_size: int, *, rngs: nnx.Rngs):
        self.kernel_size = int(kernel_size)
        self.conv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            kernel_init=nnx.initializers.normal(0.01),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        k = self.kernel_size
        x_pad = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
        return x + self.conv(x_pad)


class GatedSLiSDELayer(nnx.Module):
    """Bilinear-gated in-flow SDE layer.

    Modulates the diagonal of the inner SDE's affine transition matrix and
    adds a residual offset to the affine intercept. See module docstring
    for the full forward pipeline.
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
        norm_type: NormType = "rms",
        final_scale: float = 0.1,
        use_causal_conv: bool = False,
        causal_conv_kernel_size: int = 4,
        use_time_features: bool = True,
        time_omega: float = 6.283185307179586,  # 2π
        base_layer_kwargs=None,
        *,
        rngs: nnx.Rngs,
    ):
        if matrix_type not in ("diag", "blockdiag", "dense"):
            raise ValueError(f"unknown matrix_type={matrix_type!r}")
        if norm_type not in ("layer", "rms"):
            raise ValueError(
                f"GatedSLiSDELayer requires norm_type ∈ {{'layer','rms'}} "
                f"(got {norm_type!r})."
            )
        self.dim = dim
        self.matrix_type = matrix_type
        self.block_size = block_size
        self.final_scale = float(final_scale)
        self.norm_type = norm_type
        self.use_causal_conv = bool(use_causal_conv)
        self.use_time_features = bool(use_time_features)
        self.time_omega = float(time_omega)
        base_layer_kwargs = base_layer_kwargs or {}

        # Effective gate-input dim: norm(x_prev) + optional 4 time channels.
        gate_in_dim = dim + (TIME_AUG_DIM if self.use_time_features else 0)

        # Inner SDE produces the base ``(F, g)``.
        self.sde = SLiSDE(
            dim=dim,
            noise_dim=noise_dim,
            matrix_type=matrix_type,
            block_size=block_size,
            noise_type=noise_type,
            w_init_std=w_init_std,
            approximate_exponential=approximate_exponential,
            lora_rank=lora_rank,
            **base_layer_kwargs,
            rngs=rngs,
        )

        # Optional causal 1-D conv on x_prev (residual, near-identity init).
        self.causal_conv = (
            CausalConv1D(dim, kernel_size=causal_conv_kernel_size, rngs=rngs)
            if self.use_causal_conv else None
        )

        # Always-on normalisation of x_prev.
        self.gate_norm = (
            nnx.RMSNorm(dim, rngs=rngs)
            if norm_type == "rms"
            else nnx.LayerNorm(dim, rngs=rngs)
        )

        # Bilinear gates (small-init so the layer is ≈ unmodulated at step 0).
        small_init = dict(
            kernel_init=nnx.initializers.normal(0.01),
            bias_init=nnx.initializers.zeros,
        )
        # g-side bilinear residual offset: o = σ(W_go · x) ⊙ (W_io · x)
        self.W_go = nnx.Linear(gate_in_dim, dim, rngs=rngs, **small_init)
        self.W_io = nnx.Linear(gate_in_dim, dim, rngs=rngs, **small_init)
        # F-side diag-only bilinear scale: α = 1 + ε · σ(W_gF · x) ⊙ tanh(W_iF · x)
        self.W_gF = nnx.Linear(gate_in_dim, dim, rngs=rngs, **small_init)
        self.W_iF = nnx.Linear(gate_in_dim, dim, rngs=rngs, **small_init)

    # ------------------------------------------------------------------
    # Gate computations
    # ------------------------------------------------------------------

    def _compute_alpha(self, x_normed: Array) -> Array:
        """Per-coord scale α ∈ (1 − ε, 1 + ε), shape ``(B, T, d)``."""
        gate = jax.nn.sigmoid(self.W_gF(x_normed))
        mod = jnp.tanh(self.W_iF(x_normed))
        return 1.0 + self.final_scale * (gate * mod)

    def _compute_offset(self, x_normed: Array) -> Array:
        """Bilinear residual offset on g, shape ``(B, T, d)``."""
        return jax.nn.sigmoid(self.W_go(x_normed)) * self.W_io(x_normed)

    def _apply_alpha_to_F_diag_only(self, F: Array, alpha: Array) -> Array:
        """Multiply only the diagonal of ``F`` by ``α``; off-diagonals untouched."""
        if self.matrix_type == "diag":
            return F * alpha  # F is itself the diagonal vector
        if self.matrix_type == "blockdiag":
            B, T, nb, bs, _ = F.shape
            alpha_blk = alpha.reshape(B, T, nb, bs)
            eye = jnp.eye(bs, dtype=F.dtype)
            mult = 1.0 + (alpha_blk[..., :, None] - 1.0) * eye
            return F * mult
        # dense
        d = F.shape[-1]
        eye = jnp.eye(d, dtype=F.dtype)
        mult = 1.0 + (alpha[..., :, None] - 1.0) * eye
        return F * mult

    def _build_time_aug(self, n_steps: int, batch_size: int, dtype) -> Array:
        """Return ``(B, T, 4)`` time features ``(t, t², sin ωt, cos ωt)``."""
        tau = time_feature_vector(n_steps, self.time_omega, dtype=dtype)  # (T, 5)
        tau = tau[:, 1:]                                                   # drop constant col
        return jnp.broadcast_to(tau[None, :, :], (batch_size, n_steps, TIME_AUG_DIM))

    # ------------------------------------------------------------------
    # Forward call
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        x_prev: Array,
        n_steps,
        dt=1.0,
        batch_size=1,
        parallel_steps=1,
        z0=None,
        dW=None,
        return_path=True,
    ):
        # 0. Optional causal-conv preprocessing (residual, near-identity init).
        if self.causal_conv is not None:
            x_prev = self.causal_conv(x_prev)
        # 1. Normalise.
        x_normed = self.gate_norm(x_prev)
        # 1b. Stack deterministic time features AFTER conv + norm (keeps the
        #     conv focused on the stochastic state and preserves the natural
        #     scale of the deterministic features).
        if self.use_time_features:
            B_dyn, T_dyn = x_normed.shape[0], x_normed.shape[1]
            tau = self._build_time_aug(T_dyn, B_dyn, x_normed.dtype)
            gate_in = jnp.concatenate([x_normed, tau], axis=-1)
        else:
            gate_in = x_normed
        # 2. Base affine flows from the inner SDE.
        flows = self.sde._build_flows(jnp.asarray(dt), dW)
        F, g = flows
        # 3. F-side: diag-only bilinear scale (structure-preserving).
        F_mod = self._apply_alpha_to_F_diag_only(F, self._compute_alpha(gate_in))
        # 4. g-side: bilinear residual offset via the base layer's injector
        #    so noise-type-specific reshape (blockdiag) is respected.
        F_mod, g_mod = self.sde._inject_offset((F_mod, g), self._compute_offset(gate_in))
        # 5. Standard affine scan.
        def run_one(z0_b, F_b, g_b):
            return self.sde._run_affine_scan(z0_b, (F_b, g_b), parallel_steps)
        paths = jax.vmap(run_one)(z0, F_mod, g_mod)
        return paths if return_path else paths[:, -1, :]
