# benchmarks/neural_sde.py
"""
Neural SDE benchmark model with MLP-based drift and diffusion (Flax NNX).

Three diffusion variants:

1. **full**:        ``sigma(z, t)``  -- MLP takes ``(z, t/T)``
2. **time_only**:   ``sigma(t)``     -- MLP takes only ``(t/T,)``
3. **time_state**:  ``sigma(t) * z`` -- MLP takes ``(t/T,)``, output multiplied by z

All variants operate in a latent space of dimension ``dim``, using
Euler-Maruyama time-stepping via ``jax.lax.scan``, followed by an
output projection to ``output_dim``.

**Implementation note**: All MLP weights are created with ``nnx.Param``
and used with raw ``jnp.dot`` inside the scan body to avoid tracer leaks.
"""
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from v3.benchmarks.neural_sde_config import NeuralSDEConfig

Array = jnp.ndarray


def _get_activation_fn(activation: str):
    """Return an activation function by name."""
    if activation == "gelu":
        return jax.nn.gelu
    elif activation == "tanh":
        return jnp.tanh
    elif activation == "relu":
        return jax.nn.relu
    else:
        raise ValueError(f"Unknown activation: {activation}")


def _apply_norm(z: Array, norm_type: str, scale: Array, bias: Optional[Array],
                eps: float = 1e-6) -> Array:
    """Pure-function LayerNorm/RMSNorm (used inside the scan body).

    ``scale`` and ``bias`` are raw arrays captured outside the scan.  When
    ``norm_type`` is ``"none"`` the function is a no-op returning ``z``.
    """
    if norm_type == "rms":
        rms = jnp.sqrt(jnp.mean(z * z, axis=-1, keepdims=True) + eps)
        return (z / rms) * scale
    if norm_type == "layer":
        mean = jnp.mean(z, axis=-1, keepdims=True)
        var = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
        norm = (z - mean) / jnp.sqrt(var + eps)
        return norm * scale + (bias if bias is not None else 0.0)
    return z


def _mlp_forward_varlen(
    x: Array,
    weights: List[Tuple[Array, Array]],
    act_fn,
    use_residual: bool = False,
) -> Array:
    """Variable-depth MLP forward pass (pure function, no Flax calls)."""
    h = x
    n_layers = len(weights)
    for i, (w, b) in enumerate(weights):
        out = h @ w + b
        if i < n_layers - 1:
            out = act_fn(out)
            if use_residual and h.shape[-1] == out.shape[-1]:
                out = out + h
        h = out
    return h


class NeuralSDEStack(nnx.Module):
    """Neural SDE benchmark with MLP drift and configurable diffusion.

    Attributes:
        config: neural SDE configuration.
    """

    def __init__(self, config: NeuralSDEConfig, *, rngs: nnx.Rngs):
        cfg = config
        self.config = cfg
        self.rngs = rngs

        ki = lambda key, shape: cfg.w_init_std * jr.normal(key, shape)
        ki_out = lambda key, shape: cfg.w_init_std * 0.1 * jr.normal(key, shape)
        n_hidden = cfg.num_hidden_layers

        key = rngs.params()
        self._n_drift_layers = n_hidden + 1
        self._n_diff_layers = n_hidden + 1

        # -- Drift MLP: (dim + 1) -> [hidden]^n -> dim -----
        drift_in_dim = cfg.dim + 1
        prev_dim = drift_in_dim
        for i in range(n_hidden):
            key, k1, k2 = jr.split(key, 3)
            setattr(self, f"drift_w{i}", nnx.Param(ki(k1, (prev_dim, cfg.hidden_dim))))
            setattr(self, f"drift_b{i}", nnx.Param(jnp.zeros((cfg.hidden_dim,))))
            prev_dim = cfg.hidden_dim
        key, k1, k2 = jr.split(key, 3)
        setattr(self, f"drift_w{n_hidden}", nnx.Param(ki_out(k1, (prev_dim, cfg.dim))))
        setattr(self, f"drift_b{n_hidden}", nnx.Param(jnp.zeros((cfg.dim,))))

        # -- Diffusion MLP -----
        if cfg.diffusion_type == "full":
            diff_in_dim = cfg.dim + 1
            diff_out_dim = cfg.dim * cfg.noise_dim
        elif cfg.diffusion_type == "time_only":
            diff_in_dim = 1
            diff_out_dim = cfg.dim * cfg.noise_dim
        else:  # time_state
            diff_in_dim = 1
            diff_out_dim = cfg.dim

        prev_dim = diff_in_dim
        for i in range(n_hidden):
            key, k1, k2 = jr.split(key, 3)
            setattr(self, f"diff_w{i}", nnx.Param(ki(k1, (prev_dim, cfg.hidden_dim))))
            setattr(self, f"diff_b{i}", nnx.Param(jnp.zeros((cfg.hidden_dim,))))
            prev_dim = cfg.hidden_dim
        key, k1, k2 = jr.split(key, 3)
        setattr(self, f"diff_w{n_hidden}", nnx.Param(ki_out(k1, (prev_dim, diff_out_dim))))
        setattr(self, f"diff_b{n_hidden}", nnx.Param(jnp.zeros((diff_out_dim,))))

        # -- Output projection: dim -> output_dim -----
        key, k1, k2 = jr.split(key, 3)
        self.out_w = nnx.Param(ki_out(k1, (cfg.dim, cfg.output_dim)))
        self.out_b = nnx.Param(jnp.zeros((cfg.output_dim,)))

        # -- Optional pre-step normalization of the MLP input ----------
        # Scale is learnable (init=1), bias is learnable (init=0) only for
        # LayerNorm. RMSNorm has no bias.
        if cfg.norm_type in ("layer", "rms"):
            self.norm_scale = nnx.Param(jnp.ones((cfg.dim,)))
            if cfg.norm_type == "layer":
                self.norm_bias = nnx.Param(jnp.zeros((cfg.dim,)))
            else:
                self.norm_bias = None
        else:
            self.norm_scale = None
            self.norm_bias = None

    def __call__(
        self,
        *,
        batch_size: int,
        return_path: bool = True,
        z0: Optional[Array] = None,
        y0: Optional[Array] = None,
    ) -> Array:
        """Forward pass: sample paths via Euler-Maruyama."""
        cfg = self.config
        dt = cfg.T / cfg.n_steps
        sqrt_dt = jnp.sqrt(dt)

        act_fn = _get_activation_fn(cfg.activation)

        if z0 is None:
            key0 = self.rngs.init()
            z0 = jr.normal(key0, (batch_size, cfg.dim))

        noise_key = self.rngs.noise()
        dW = sqrt_dt * jr.normal(noise_key, (batch_size, cfg.n_steps, cfg.noise_dim))

        t_grid = jnp.linspace(0.0, 1.0, cfg.n_steps, endpoint=False)

        # Capture raw arrays for scan closure
        drift_weights = [
            (getattr(self, f"drift_w{i}")[...], getattr(self, f"drift_b{i}")[...])
            for i in range(self._n_drift_layers)
        ]
        diff_weights = [
            (getattr(self, f"diff_w{i}")[...], getattr(self, f"diff_b{i}")[...])
            for i in range(self._n_diff_layers)
        ]
        out_w = self.out_w[...]
        out_b = self.out_b[...]

        dim = cfg.dim
        noise_dim = cfg.noise_dim
        diffusion_type = cfg.diffusion_type
        use_residual = cfg.use_residual
        drift_scale = cfg.drift_scale
        diffusion_scale = cfg.diffusion_scale
        norm_type = cfg.norm_type
        norm_scale = self.norm_scale[...] if self.norm_scale is not None else None
        norm_bias = self.norm_bias[...] if self.norm_bias is not None else None

        def em_step(z, inputs):
            t_norm, dW_k = inputs
            # Normalize the MLP input (not the SDE state itself) when enabled.
            z_feat = _apply_norm(z, norm_type, norm_scale, norm_bias) \
                if norm_type != "none" else z

            drift_in = jnp.concatenate([z_feat, t_norm[None]])
            mu = _mlp_forward_varlen(drift_in, drift_weights, act_fn,
                                     use_residual) * drift_scale

            if diffusion_type == "full":
                diff_in = jnp.concatenate([z_feat, t_norm[None]])
                sigma_flat = _mlp_forward_varlen(
                    diff_in, diff_weights, act_fn, use_residual
                ) * diffusion_scale
                sigma = sigma_flat.reshape(dim, noise_dim)
                noise_term = sigma @ dW_k
            elif diffusion_type == "time_only":
                diff_in = t_norm[None]
                sigma_flat = _mlp_forward_varlen(
                    diff_in, diff_weights, act_fn, use_residual
                ) * diffusion_scale
                sigma = sigma_flat.reshape(dim, noise_dim)
                noise_term = sigma @ dW_k
            else:  # time_state
                diff_in = t_norm[None]
                sigma_vec = _mlp_forward_varlen(
                    diff_in, diff_weights, act_fn, use_residual
                ) * diffusion_scale
                dW_agg = jnp.sum(dW_k) / jnp.sqrt(jnp.float32(noise_dim))
                noise_term = sigma_vec * z * dW_agg

            z_next = z + mu * dt + noise_term
            return z_next, z_next

        def scan_one_path(z0_b, dW_b):
            scan_inputs = (t_grid, dW_b)
            _, zs = jax.lax.scan(em_step, z0_b, scan_inputs)
            return jnp.concatenate([z0_b[None, :], zs], axis=0)

        paths_latent = jax.vmap(scan_one_path)(z0, dW)

        output = paths_latent @ out_w + out_b

        if y0 is not None:
            shift = y0[None, None, :] - output[:, 0:1, :]
            output = output + shift

        return output if return_path else output[:, -1, :]
