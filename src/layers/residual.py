from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from .base import SLiSDE
from ..utils import apply_activation

Array = jnp.ndarray
MatrixType = Literal["diag", "blockdiag", "dense"]
NoiseType = Literal["multiplicative", "additive", "general"]
ActivationType = Literal["tanh", "gelu", "glu"]
NormType = Literal["none", "layer", "rms"]


class ResidualLayer(nnx.Module):
    """
    SLiSDE wrapped with normalization, gated projection, and activation.

    Computes:
        z = SLiSDE(z0, dW)
        z_normed = Norm(z)                           (optional)
        activated = activation(proj(z_normed))
        output = sigmoid(gate) * activated

    The residual connection h += output is done at the model level.
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
        activation: ActivationType = "tanh",
        base_layer_kwargs=None,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.activation = activation
        self.norm_type = norm_type
        base_layer_kwargs = base_layer_kwargs or {}

        # Inner SDE (base SLiSDE; time dependence controlled via base_layer_kwargs).
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

        # Normalization
        if norm_type == "rms":
            self.norm = nnx.RMSNorm(dim, rngs=rngs)
        elif norm_type == "layer":
            self.norm = nnx.LayerNorm(dim, rngs=rngs)
        else:
            self.norm = None

        # Gate and projection.
        # (A3) Initialise ``res_gate`` so that ``sigmoid(res_gate) ≈ 1`` at
        # start, instead of the old ``0.5``.  This prevents every residual
        # branch from being halved at init, which compounded badly with the
        # other zero-init modulations in the stack (time-gate deltas,
        # ``L_mix`` in the Girsanov head, etc.).  logit(1 - 1e-4) ≈ 9.21.
        self.res_gate = nnx.Param(jnp.full((dim,), 9.21, dtype=jnp.float32))
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        if activation == "glu":
            self.proj_glu = nnx.Linear(dim, dim, rngs=rngs)
        else:
            self.proj_glu = None

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
    ):
        """Run SDE, normalise, activate, and apply residual gate."""
        z = self.sde(
            n_steps=n_steps,
            dt=dt,
            batch_size=batch_size,
            parallel_steps=parallel_steps,
            z0=z0,
            dW=dW,
            return_path=True,
        )
        proj_input = self.norm(z) if self.norm is not None else z
        activated = apply_activation(proj_input, self.activation, self.proj, self.proj_glu)
        gate = jax.nn.sigmoid(self.res_gate[...])
        output = gate[None, None, :] * activated
        return output if return_path else output[:, -1, :]
