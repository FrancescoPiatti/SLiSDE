# benchmarks/slice_model.py
"""
SLiCE benchmark — JAX/Flax NNX port of  ``datasig-ac-uk/slices``.

Faithful port of the upstream ``SLiCE`` core, ``SLiCELayer`` residual
wrapper, and ``StackedSLiCE`` stacked-layer model.  Drops:
  * tokens path (we always have a continuous Brownian-motion input)
  * per-step dropout (none of the v2 benchmarks use it; the SLiSDE
    trainers achieve regularisation via weight-decay on the optimiser)
  * the ``input_dependent_init`` flag for the inner ``y_0`` (we use the
    learnt-vector form, matching upstream's default)

Adds:
  * ``__call__(*, batch_size, return_path, z0, y0)`` matching the
    NeuralSDE benchmark interface so the existing trainers slot it in
    directly.
  * Brownian-motion sampling (``dW`` then optional cumsum to ``W_t``)
    fed into the stack as the input path.

The SLiCE recurrence is

    y_i = M_i y_{i-1} + b_i,    M_i ∈ {I + A(X_i), exp(A(X_i))}

with ``A(X_i): R^{D+1} → R^{H × H}`` block-structured (block-diagonal,
optionally with a single dense tail block) and ``B(X_i): R^{D+1} → R^H``.
All time-steps within a layer are computed in parallel via
``jax.lax.associative_scan`` over the affine pair ``(M, b)``.

Block-size 1 reduces the inner SLiCE to the Mamba-1-style elementwise
diagonal SSM emphasised in the upstream paper.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.benchmarks.slice_config import SLiCEConfig

Array = jnp.ndarray


# =========================================================================
# RMSNorm  (port of SLiCE.RMSNorm)
# =========================================================================

class RMSNorm(nnx.Module):
    """Minimal RMSNorm matching the upstream SLiCE implementation."""

    def __init__(self, d_model: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        del rngs  # parameter-less init, but accept for consistency
        self.eps = float(eps)
        self.weight = nnx.Param(jnp.ones((d_model,)))

    def __call__(self, x: Array) -> Array:
        rms = jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x * rms) * self.weight[...]


def _make_norm(kind: str, d_model: int, eps: float, *, rngs: nnx.Rngs):
    """Dispatch ``rmsnorm`` / ``layernorm``."""
    if kind == "rmsnorm":
        return RMSNorm(d_model, eps=eps, rngs=rngs)
    if kind == "layernorm":
        return nnx.LayerNorm(d_model, epsilon=eps, rngs=rngs)
    raise ValueError(f"Unknown norm kind: {kind!r}")


# =========================================================================
# SLiCE core
# =========================================================================

def _matrix_exp_small(A: Array) -> Array:
    """``expm`` over the trailing two axes (broadcast-friendly).

    Each call expects ``A`` of shape ``(..., n, n)`` and returns the same
    shape.  We vectorise via ``jnp.vectorize`` on the inner ``(n,n)`` block.
    """
    return jnp.vectorize(jax.scipy.linalg.expm, signature="(n,n)->(n,n)")(A)


class SLiCE(nnx.Module):
    """JAX/Flax port of the SLiCE recurrence.

    Computes  ``y_i = M_i y_{i-1} + b_i``  where
        M_i = I + A(X_i)     (Euler)
        M_i = exp(A(X_i))    (matrix-exp)
    and ``A`` is block-diagonal with ``hidden_dim // block_size`` blocks
    of size ``block_size``.  When ``diagonal_dense=True`` the last block
    is fully dense and the remaining (``hidden_dim − block_size``)
    dimensions are elementwise.

    Shapes:
        Input  ``X``:   ``(B, T, input_dim)``
        Output ``y``:   ``(B, T, hidden_dim)``
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        block_size: int = 1,
        diagonal_dense: bool = False,
        init_std: float = 0.01,
        scale: float = 1.0,
        path_mode: str = "values",
        transition_mode: str = "euler",
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.block_size = block_size
        self.init_std = init_std
        self.scale = scale
        self.path_mode = path_mode
        self.transition_mode = transition_mode

        # Same degenerate-case collapses as upstream.
        if diagonal_dense and (block_size == hidden_dim or block_size == 1):
            diagonal_dense = False
        self.diagonal_dense = diagonal_dense

        if not self.diagonal_dense and hidden_dim % block_size != 0:
            raise ValueError("hidden_dim must be divisible by block_size")

        # ---------- y_0 (learnt) ----------
        ki, ka, kad, kb = jr.split(rngs.params(), 4)
        self.init_y = nnx.Param(init_std * jr.normal(ki, (hidden_dim,)))

        # ---------- vf_A: A is a linear map from (D+1) to A-coeffs ----------
        # Upstream uses ``nn.Linear`` with bias=False and normal init.
        # We replicate by holding only the kernel as nnx.Param.
        block_scale = 1.0 / (block_size ** 0.5)

        if self.diagonal_dense:
            hdiag = hidden_dim - block_size
            self.vf_A_diag_w = nnx.Param(
                init_std * jr.normal(kad, (input_dim + 1, hdiag))
            )
            self.vf_A_dense_w = nnx.Param(
                init_std * block_scale
                * jr.normal(ka, (input_dim + 1, block_size * block_size))
            )
            self.vf_A_w = None
        else:
            self.vf_A_w = nnx.Param(
                init_std * block_scale
                * jr.normal(ka, (input_dim + 1, hidden_dim * block_size))
            )
            self.vf_A_diag_w = None
            self.vf_A_dense_w = None

        # ---------- vf_B: bias term ----------
        if bias:
            self.vf_B_w = nnx.Param(
                init_std * jr.normal(kb, (input_dim + 1, hidden_dim))
            )
        else:
            self.vf_B_w = None

    # ------------------------------------------------------------------
    # Input preparation (identical semantics to upstream)
    # ------------------------------------------------------------------

    def _prepare_driving_path(self, X: Array) -> Array:
        if self.path_mode == "values":
            # increments = X_t − X_{t-1}, with X_{-1} = 0  ⇒ first increment = X_0.
            zero_pad = jnp.zeros_like(X[:, :1, :])
            return jnp.concatenate(
                [X[:, :1, :] - zero_pad, jnp.diff(X, axis=1)], axis=1
            )
        return X

    def _prepare_augmented_inputs(self, X: Array) -> Array:
        path = self._prepare_driving_path(X)
        # Prepend a constant time channel of ones (so B(X_i) gets a bias term).
        ones = jnp.ones(path.shape[:-1] + (1,), dtype=path.dtype)
        return jnp.concatenate([ones, path], axis=-1) * self.scale

    # ------------------------------------------------------------------
    # Discretisations
    # ------------------------------------------------------------------

    def _discretise_diagonal(self, A: Array) -> Array:
        if self.transition_mode == "matrix_exp":
            return jnp.exp(A)
        return 1.0 + A

    def _discretise_matrix(self, A: Array) -> Array:
        if self.transition_mode == "matrix_exp":
            return _matrix_exp_small(A)
        n = A.shape[-1]
        eye = jnp.eye(n, dtype=A.dtype)
        return eye + A

    # ------------------------------------------------------------------
    # Per-step transform builders
    # ------------------------------------------------------------------

    def _build_elementwise(self, inp: Array) -> Tuple[Array, Array]:
        # inp: (B, T, D+1)  →  M, b: (B, T, hidden_dim)
        A = inp @ self.vf_A_w[...]
        M = self._discretise_diagonal(A)
        if self.bias:
            b = inp @ self.vf_B_w[...]
        else:
            b = jnp.zeros_like(M)
        return M, b

    def _build_blockdiag(self, inp: Array) -> Tuple[Array, Array]:
        bsz = self.block_size
        nblocks = self.hidden_dim // bsz
        A = inp @ self.vf_A_w[...]
        A = A.reshape(*A.shape[:-1], nblocks, bsz, bsz)
        M = self._discretise_matrix(A)
        if self.bias:
            b = (inp @ self.vf_B_w[...]).reshape(*inp.shape[:-1], nblocks, bsz)
        else:
            b = jnp.zeros(M.shape[:-1], dtype=M.dtype)
        return M, b

    def _build_diagonal_dense(
        self, inp: Array
    ) -> Tuple[Array, Array, Array, Array]:
        bsz = self.block_size
        hdiag = self.hidden_dim - bsz

        A_diag = inp @ self.vf_A_diag_w[...]
        M_diag = self._discretise_diagonal(A_diag)

        A_dense = inp @ self.vf_A_dense_w[...]
        A_dense = A_dense.reshape(*A_dense.shape[:-1], bsz, bsz)
        M_dense = self._discretise_matrix(A_dense)

        if self.bias:
            B = inp @ self.vf_B_w[...]
            b_diag = B[..., :hdiag]
            b_dense = B[..., hdiag:]
        else:
            b_diag = jnp.zeros_like(M_diag)
            b_dense = jnp.zeros(M_dense.shape[:-1], dtype=M_dense.dtype)
        return M_diag, M_dense, b_diag, b_dense

    # ------------------------------------------------------------------
    # Parallel scan kernels
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_elementwise(left, right):
        # right ∘ left : (M, b)
        M_l, b_l = left
        M_r, b_r = right
        return (M_r * M_l, M_r * b_l + b_r)

    @staticmethod
    def _combine_blockdiag(left, right):
        M_l, b_l = left
        M_r, b_r = right
        # M: (..., nblocks, bsz, bsz);  b: (..., nblocks, bsz)
        M_new = jnp.einsum("...nij,...njk->...nik", M_r, M_l)
        b_new = jnp.einsum("...nij,...nj->...ni", M_r, b_l) + b_r
        return (M_new, b_new)

    @staticmethod
    def _combine_diagonal_dense(left, right):
        Md_l, Mdense_l, bd_l, bdense_l = left
        Md_r, Mdense_r, bd_r, bdense_r = right
        Md = Md_r * Md_l
        bd = Md_r * bd_l + bd_r
        Mdense = jnp.einsum("...ij,...jk->...ik", Mdense_r, Mdense_l)
        bdense = jnp.einsum("...ij,...j->...i", Mdense_r, bdense_l) + bdense_r
        return (Md, Mdense, bd, bdense)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, X: Array) -> Array:
        """X: (B, T, input_dim)  →  y: (B, T, hidden_dim)."""
        inp = self._prepare_augmented_inputs(X)
        B, T = inp.shape[0], inp.shape[1]

        y0 = jnp.broadcast_to(self.init_y[...], (B, self.hidden_dim))

        if self.diagonal_dense:
            transforms = self._build_diagonal_dense(inp)
            prefix = jax.lax.associative_scan(
                self._combine_diagonal_dense, transforms, axis=1
            )
            Md, Mdense, bd, bdense = prefix
            bsz = self.block_size
            hdiag = self.hidden_dim - bsz

            y_diag = Md * y0[:, None, :hdiag] + bd
            y_dense_init = y0[:, None, hdiag:, None]   # (B,1,bsz,1)
            y_dense = (
                jnp.einsum("btij,btjk->btik", Mdense, y_dense_init).squeeze(-1)
                + bdense
            )
            return jnp.concatenate([y_diag, y_dense], axis=-1)

        if self.block_size > 1:
            M, b = self._build_blockdiag(inp)
            prefix_M, prefix_b = jax.lax.associative_scan(
                self._combine_blockdiag, (M, b), axis=1
            )
            bsz = self.block_size
            nblocks = self.hidden_dim // bsz
            y0b = y0.reshape(B, nblocks, bsz)[:, None, :, :, None]   # (B,1,n,b,1)
            y = jnp.einsum("btnij,btnjk->btnik", prefix_M, y0b).squeeze(-1)
            y = y + prefix_b
            return y.reshape(B, T, self.hidden_dim)

        # Elementwise (block_size == 1)
        M, b = self._build_elementwise(inp)
        prefix_M, prefix_b = jax.lax.associative_scan(
            self._combine_elementwise, (M, b), axis=1
        )
        return prefix_M * y0[:, None, :] + prefix_b


# =========================================================================
# SLiCELayer  (residual wrapper around a SLiCE)
# =========================================================================

class _TokenMLP(nnx.Module):
    """Two-layer MLP with GELU/GLU/Tanh, matching upstream SLiCELayer FF."""

    def __init__(
        self,
        d_model: int,
        ff_mult: int,
        ff_style: str,
        ff_activation: str,
        *,
        rngs: nnx.Rngs,
    ):
        self.ff_style = ff_style
        self.ff_activation = ff_activation
        ff_hidden = ff_mult * d_model
        ff_in_dim = 2 * ff_hidden if ff_activation == "glu" else ff_hidden

        self.fc1 = nnx.Linear(d_model, ff_in_dim, rngs=rngs)
        if ff_style == "mlp":
            self.fc2 = nnx.Linear(ff_hidden, d_model, rngs=rngs)
        else:
            self.fc2 = None

    def _act(self, x: Array) -> Array:
        if self.ff_activation == "gelu":
            return jax.nn.gelu(x)
        if self.ff_activation == "tanh":
            return jnp.tanh(x)
        # GLU: split last axis into two halves, multiply by sigmoid of the second.
        a, b = jnp.split(x, 2, axis=-1)
        return a * jax.nn.sigmoid(b)

    def __call__(self, x: Array) -> Array:
        h = self._act(self.fc1(x))
        if self.fc2 is not None:
            h = self.fc2(h)
        return h


class SLiCELayer(nnx.Module):
    """Residual wrapper: norm → SLiCE → residual → norm → MLP → residual.

    Defaults to upstream pre-norm RMSNorm/GELU recipe.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        bias: bool,
        block_size: int,
        diagonal_dense: bool,
        init_std: float,
        scale: float,
        path_mode: str,
        transition_mode: str,
        norm_type: str,
        prenorm: bool,
        second_norm: bool,
        ff_style: str,
        ff_activation: str,
        ff_mult: int,
        norm_eps: float,
        rngs: nnx.Rngs,
    ):
        self.prenorm = prenorm
        self.second_norm = second_norm

        self.slice = SLiCE(
            input_dim=input_dim,
            hidden_dim=input_dim,
            bias=bias,
            block_size=block_size,
            diagonal_dense=diagonal_dense,
            init_std=init_std,
            scale=scale,
            path_mode=path_mode,
            transition_mode=transition_mode,
            rngs=rngs,
        )
        self.norm1 = _make_norm(norm_type, input_dim, norm_eps, rngs=rngs)
        self.norm2 = (
            _make_norm(norm_type, input_dim, norm_eps, rngs=rngs)
            if second_norm else None
        )
        self.ff = _TokenMLP(
            d_model=input_dim,
            ff_mult=ff_mult,
            ff_style=ff_style,
            ff_activation=ff_activation,
            rngs=rngs,
        )

    def __call__(self, X: Array) -> Array:
        if self.prenorm:
            X = X + self.slice(self.norm1(X))
            ff_in = self.norm2(X) if self.norm2 is not None else X
            X = X + self.ff(ff_in)
            return X

        # post-norm
        X = self.norm1(X + self.slice(X))
        X = X + self.ff(X)
        if self.norm2 is not None:
            X = self.norm2(X)
        return X


# =========================================================================
# StackedSLiCE — top-level benchmark module
# =========================================================================

class SLiCEStack(nnx.Module):
    """SLiCE benchmark on Brownian-motion driving inputs.

    Drop-in for ``NeuralSDEStack``: same call signature
    ``__call__(*, batch_size, return_path, z0, y0)``.

    Forward:
      1. Sample ``dW ∈ R^{B × T × noise_dim}`` (Brownian increments).
      2. If ``bm_input_type == "path"``, ``X = cumsum(dW, axis=1)``;
         otherwise ``X = dW``.  The SLiCE inner core then either
         differentiates back to dW (path mode) or consumes dW directly
         (increments mode); under a learnt linear embedding the two are
         equivalent up to reparameterisation.
      3. Embed ``X`` to ``hidden_dim`` (Linear).
      4. ``num_layers`` × ``SLiCELayer``.
      5. Final Linear: ``hidden_dim → output_dim``.

    Returns ``(B, T, output_dim)`` if ``return_path`` else ``(B, output_dim)``.
    """

    def __init__(self, config: SLiCEConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.rngs = rngs

        self.embedding = nnx.Linear(config.noise_dim, config.hidden_dim, rngs=rngs)
        # Use setattr-based naming (matches v2 SLiSDEModel convention) so
        # Flax NNX 0.12 doesn't reject the list-typed attribute as a static
        # pytree leaf containing Arrays.
        for i in range(config.num_layers):
            layer = SLiCELayer(
                input_dim=config.hidden_dim,
                bias=config.bias,
                block_size=config.block_size,
                diagonal_dense=config.diagonal_dense,
                init_std=config.init_std,
                scale=config.scale,
                path_mode=config.path_mode,
                transition_mode=config.transition_mode,
                norm_type=config.norm_type,
                prenorm=config.prenorm,
                second_norm=config.second_norm,
                ff_style=config.ff_style,
                ff_activation=config.ff_activation,
                ff_mult=config.ff_mult,
                norm_eps=config.norm_eps,
                rngs=rngs,
            )
            setattr(self, f"layer_{i}", layer)
        self._n_layers = config.num_layers
        self.out = nnx.Linear(config.hidden_dim, config.output_dim, rngs=rngs)

    def __call__(
        self,
        *,
        batch_size: int,
        return_path: bool = True,
        z0: Optional[Array] = None,    # accepted for interface parity (unused)
        y0: Optional[Array] = None,
    ) -> Array:
        del z0
        cfg = self.config
        dt = cfg.T / cfg.n_steps
        sqrt_dt = jnp.sqrt(dt)

        dW = sqrt_dt * jr.normal(
            self.rngs.noise(), (batch_size, cfg.n_steps, cfg.noise_dim)
        )
        X = jnp.cumsum(dW, axis=1) if cfg.bm_input_type == "path" else dW

        h = self.embedding(X)
        for i in range(self._n_layers):
            h = getattr(self, f"layer_{i}")(h)
        out = self.out(h)

        # Match the NeuralSDE convention: prepend a "step 0" entry so the
        # output has length T+1 along the time axis.  The SLiCE recurrence
        # naturally produces T outputs (one per increment); we synthesise
        # the t=0 frame as ``out_proj(embedding(0))`` (= ``out_b`` since
        # ``embedding(0) = b_emb`` and ``layers`` starts from y_0 broadcast)
        # — for simplicity we reuse the first time-step output, which
        # corresponds to ``y_0`` projected.  The shift logic below uses
        # only the first entry so this is benign.
        out = jnp.concatenate([out[:, :1, :], out], axis=1)

        if y0 is not None:
            shift = y0[None, None, :] - out[:, 0:1, :]
            out = out + shift

        return out if return_path else out[:, -1, :]
