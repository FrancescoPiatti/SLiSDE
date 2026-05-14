# src/structured_matrices.py
"""
Structured matrix construction helpers for Flax NNX.

Supported structured types:
  - diag:      diagonal matrix stored as vector (d,)
  - blockdiag: block-diagonal stored as (nb, bs, bs) where d = nb * bs
  - dense:     full dense matrix stored as (d, d)

This file contains ONLY parameter creation logic.
Flow construction, combine rules, and scan logic live in flows.py.

All functions return raw JAX arrays. The caller is responsible for
wrapping them in nnx.Param when assigning to module attributes.
"""
from typing import Literal, Union

import jax
import jax.numpy as jnp
import jax.random as jr

Array = jnp.ndarray
MatrixType = Literal["diag", "blockdiag", "dense"]


def create_drift_params(
    key: jax.Array,
    matrix_type: MatrixType,
    dim: int,
    block_size: int,
    w_init_std: float,
) -> Array:
    """Create drift matrix A parameters in the chosen structured form.

    Returns:
        diag      -> vector (d,)
        blockdiag -> tensor (nb, bs, bs)
        dense     -> matrix (d, d)
    """
    bs = block_size

    if matrix_type == "diag":
        return w_init_std * jr.normal(key, (dim,))

    if matrix_type == "blockdiag":
        nb = dim // bs
        std = w_init_std / jnp.sqrt(bs)
        return std * jr.normal(key, (nb, bs, bs))

    if matrix_type == "dense":
        std = w_init_std / jnp.sqrt(dim)
        return std * jr.normal(key, (dim, dim))

    raise ValueError(f"Unknown matrix_type: {matrix_type}")


def create_diffusion_params(
    key: jax.Array,
    matrix_type: MatrixType,
    dim: int,
    noise_dim: int,
    block_size: int,
    w_init_std: float,
) -> Array:
    """Create structured diffusion parameters (C^k or D^k).

    Returns:
        diag      -> (m, d)
        blockdiag -> (m, nb, bs, bs)
        dense     -> (m, d, d)
    """
    bs = block_size
    m = noise_dim

    if matrix_type == "diag":
        return w_init_std * jr.normal(key, (m, dim))

    if matrix_type == "blockdiag":
        nb = dim // bs
        std = w_init_std / jnp.sqrt(bs)
        return std * jr.normal(key, (m, nb, bs, bs))

    if matrix_type == "dense":
        std = w_init_std / jnp.sqrt(dim)
        return std * jr.normal(key, (m, dim, dim))

    raise ValueError(f"Unknown matrix_type: {matrix_type}")


def create_dense_diffusion_params(
    key: jax.Array,
    dim: int,
    noise_dim: int,
    w_init_std: float,
) -> Array:
    """Create a dense (unstructured) additive diffusion matrix D.

    Returns:
        Dense D matrix with shape (noise_dim, dim).
    """
    return w_init_std * jr.normal(key, (noise_dim, dim))


def create_lora_diffusion_params(
    key: jax.Array,
    dim: int,
    noise_dim: int,
    lora_rank: int,
    w_init_std: float,
):
    """Create low-rank additive diffusion D = D_A @ D_B.

    D_A: (noise_dim, lora_rank)  — initialised with scaled normal
    D_B: (lora_rank, dim)        — initialised to zero (LoRA convention)

    Returns:
        (D_A, D_B) tuple of arrays.
    """
    k1, k2 = jr.split(key)
    D_A = w_init_std * jr.normal(k1, (noise_dim, lora_rank))
    D_B = jnp.zeros((lora_rank, dim))
    return D_A, D_B


def create_bias_param(key: jax.Array, dim: int) -> Array:
    """Create the bias vector b (zero-initialised), shape (d,)."""
    return jnp.zeros((dim,))
