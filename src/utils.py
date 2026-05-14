# v3/src/utils.py
"""Shared utility helpers for the SLiSDE framework.

* :func:`apply_activation` — dense-projection + activation in one call,
  used by every gated layer.
* :func:`time_feature_vector` — deterministic 5-feature time embedding
  ``(1, t, t², sin(ωt), cos(ωt))`` used by both the time-dependent base
  layer and the gated-layer time-feature augmentation.
"""
from typing import Optional

import jax
import jax.numpy as jnp

Array = jnp.ndarray

# Width of the time-feature vector returned by ``time_feature_vector``.
TIME_FEATURE_DIM = 5


def apply_activation(x: Array, activation: str, proj, proj_glu=None) -> Array:
    """Apply ``proj`` to ``x``, then a non-linearity selected by name.

    Args:
        x: Input tensor.
        activation: One of ``"tanh"``, ``"gelu"``, ``"glu"``.
        proj: ``nnx.Linear`` consuming ``x`` and producing the activation
            argument.
        proj_glu: Second ``nnx.Linear`` (same shape as ``proj``) consumed by
            the GLU gate. Required if ``activation == "glu"``.

    Returns:
        Projected, activated tensor with the same leading shape as ``x``.
    """
    if activation == "tanh":
        return jnp.tanh(proj(x))
    if activation == "gelu":
        return jax.nn.gelu(proj(x))
    # glu
    return jax.nn.sigmoid(proj_glu(x)) * proj(x)


def time_feature_vector(
    n_steps: int, omega: float, dtype=jnp.float32,
) -> Array:
    """Return the deterministic time-feature matrix ``τ(t)``.

    Each row at time index ``k`` is ``(1, t_k, t_k², sin(ωt_k), cos(ωt_k))``
    with ``t_k = (k + 0.5) / n_steps`` (mid-cell). Used both by the
    time-dependent base coefficient decoder and by gated layers that
    concatenate ``(t, t², sin ωt, cos ωt)`` to the gate input.

    Returns: array of shape ``(n_steps, TIME_FEATURE_DIM)``.
    """
    t = (jnp.arange(n_steps, dtype=dtype) + 0.5) / jnp.asarray(n_steps, dtype=dtype)
    ones = jnp.ones_like(t)
    w = jnp.asarray(omega, dtype=dtype)
    return jnp.stack([ones, t, t * t, jnp.sin(w * t), jnp.cos(w * t)], axis=-1)
