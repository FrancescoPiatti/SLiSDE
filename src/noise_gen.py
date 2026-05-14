# version3/src/noise_gen.py
"""
Brownian increment sampling for one or multiple SDE layers.

Convention:
  - Discretise [t0, T] into n_steps increments (dt = T / n_steps).
  - Time grid has n_steps + 1 points (includes t0).
  - dW has length n_steps along the increment axis.

Output shapes:
  - num_blocks == 1: (batch_size, n_steps, noise_dim)
  - num_blocks >  1: (batch_size, num_blocks, n_steps, noise_dim)

Modes:
  - "shared"      : same dW for every block
  - "independent" : independent dW per block
  - "correlated"  : dW^l = rho * dW^0 + sqrt(1-rho^2) * eps^l
"""
from functools import partial
from typing import Literal, Union

import jax
import jax.numpy as jnp
import jax.random as jr

NoiseSharing = Literal["shared", "independent", "correlated"]


# ---------------------------------------------------------------------------
# Module-level JIT-compiled sampling functions
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(2, 3, 4))
def _sample_single(key, T, batch_size, n_steps, noise_dim):
    """
    Single-block sampling (all modes collapse to the same operation).
    """
    dt = T / n_steps
    return jnp.sqrt(dt) * jr.normal(key, (batch_size, n_steps, noise_dim))


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _sample_shared(key, T, batch_size, n_steps, noise_dim, num_blocks):
    """
    Multi-block shared: one draw broadcast to all blocks.
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    dW0 = sqrt_dt * jr.normal(key, (batch_size, n_steps, noise_dim))
    return jnp.broadcast_to(
        dW0[:, None, ...],
        (batch_size, num_blocks, n_steps, noise_dim),
    )


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _sample_independent(key, T, batch_size, n_steps, noise_dim, num_blocks):
    """
    Multi-block independent: fully independent draws per block.
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    return sqrt_dt * jr.normal(key, (batch_size, num_blocks, n_steps, noise_dim))


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _sample_correlated(key, T, batch_size, n_steps, noise_dim, num_blocks, rho):
    """
    Multi-block correlated: one-factor coupling dW = rho*dW0 + sqrt(1-rho^2)*eps.
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    k0, k1 = jr.split(key, 2)
    dW0 = sqrt_dt * jr.normal(k0, (batch_size, n_steps, noise_dim))
    eps = sqrt_dt * jr.normal(k1, (batch_size, num_blocks, n_steps, noise_dim))
    rho_arr = jnp.asarray(rho, dtype=dW0.dtype)
    return rho_arr * dW0[:, None, ...] + jnp.sqrt(1.0 - rho_arr**2) * eps


# ---------------------------------------------------------------------------

class NoiseGenerator:
    """
    Samples Brownian increments dW for one or multiple SDE blocks.

    Convention:
      - num_blocks == 1: (batch_size, n_steps, noise_dim)
      - num_blocks >  1: (batch_size, num_blocks, n_steps, noise_dim)

    Modes:
      - "shared"      : same dW for every block
      - "independent" : independent dW per block
      - "correlated"  : dW^l = rho * dW^0 + sqrt(1-rho^2) * eps^l
    """

    _VALID_MODES = {"shared", "independent", "correlated"}

    def __init__(self, mode: NoiseSharing = "shared", rho: float = 0.9):
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.rho = float(rho)


    @staticmethod
    def infer_dt(*, T: Union[float, jax.Array], n_steps: int) -> jax.Array:
        """
        Infers dt from final time T and number of increments n_steps.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        T = jnp.asarray(T)
        dt = T / jnp.asarray(n_steps, dtype=T.dtype)
        return dt


    def sample(
        self,
        key: jax.Array,
        *,
        n_steps: int,
        noise_dim: int,
        T: Union[float, jax.Array],
        num_blocks: int = 1,
        batch_size: int = 1,
        augment_with_time: bool = False,
    ) -> jax.Array:
        """
        Samples Brownian increments according to the configured sharing mode.

        If ``augment_with_time`` is True, the final channel of the returned
        increment tensor is replaced with a deterministic time channel equal
        to ``dt`` at every step — this is the discrete-time equivalent of
        using ``X_t = (t, W_t)`` as the driving path.  The caller is
        responsible for setting ``noise_dim`` to include this extra channel
        (i.e. one larger than the number of pure Brownian channels).
        """
        T_arr = jnp.asarray(T)

        if augment_with_time:
            if noise_dim < 2:
                raise ValueError(
                    "augment_with_time=True requires noise_dim >= 2 "
                    "(one channel is reserved for the deterministic time channel)."
                )
            # Sample noise_dim - 1 Brownian channels then append the time channel.
            brown_dim = noise_dim - 1
        else:
            brown_dim = noise_dim

        if num_blocks == 1:
            dW = _sample_single(key, T_arr, batch_size, n_steps, brown_dim)
        elif self.mode == "shared":
            dW = _sample_shared(key, T_arr, batch_size, n_steps, brown_dim, num_blocks)
        elif self.mode == "independent":
            dW = _sample_independent(key, T_arr, batch_size, n_steps, brown_dim, num_blocks)
        else:  # correlated
            dW = _sample_correlated(
                key, T_arr, batch_size, n_steps, brown_dim, num_blocks, self.rho
            )

        if not augment_with_time:
            return dW

        # Append a deterministic time channel filled with dt.
        dt = T_arr / jnp.asarray(n_steps, dtype=T_arr.dtype)
        time_shape = dW.shape[:-1] + (1,)
        time_channel = jnp.full(time_shape, dt, dtype=dW.dtype)
        return jnp.concatenate([dW, time_channel], axis=-1)

    def sample_lastlayer_orth(
        self,
        key: jax.Array,
        *,
        n_steps: int,
        noise_dim: int,
        T: Union[float, jax.Array],
        batch_size: int = 1,
        augment_with_time: bool = False,
    ) -> jax.Array:
        """Sample a fresh orthogonal increment ``eps_tilde`` for the
        correlated last-layer Brownian motion used by the Girsanov tilt.

        Returns shape ``(batch_size, n_steps, noise_dim)``.

        Independent of the prefix sharing mode: this draws a fresh Gaussian
        increment with variance ``dt`` on every Brownian channel.  When
        ``augment_with_time`` is True the trailing channel is reserved for
        the deterministic time feature and is set to **zero** here — the
        model is responsible for restoring the ``dt`` time channel on the
        SDE-driver side after mixing.  Setting it to zero (rather than
        ``dt``) ensures that the random part of the orthogonal increment
        is well-defined as a standard Brownian increment on the
        Brownian channels alone.
        """
        T_arr = jnp.asarray(T)
        if augment_with_time:
            if noise_dim < 2:
                raise ValueError(
                    "augment_with_time=True requires noise_dim >= 2 "
                    "(one channel is reserved for the deterministic time channel)."
                )
            brown_dim = noise_dim - 1
        else:
            brown_dim = noise_dim
        eps = _sample_single(key, T_arr, batch_size, n_steps, brown_dim)
        if not augment_with_time:
            return eps
        # Pad with a zero time channel — the model overrides this to dt
        # when constructing the last-layer SDE driver.
        zero_shape = eps.shape[:-1] + (1,)
        zero_channel = jnp.zeros(zero_shape, dtype=eps.dtype)
        return jnp.concatenate([eps, zero_channel], axis=-1)
