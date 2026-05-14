# v2/src/girsanov/drift_tilt_utils.py
"""
Utility functions for v2 Girsanov drift-tilt controllers.

Provides:
  - FFT-based causal convolution tilt
  - State-space controller affine-flow construction & parallel scan
  - Radon-Nikodym log-weight computation
  - Brownian-increment tilting
"""
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


# =========================================================================
# 1. FFT tilt
# =========================================================================

def compute_causal_convolution_tilt(
    dW_P: Array,
    kernel: Array,
    c_bias: Optional[Array] = None,
) -> Array:
    """
    Compute control u_k via causal convolution on Brownian increments.

        u_k = sum_{j=0}^{k-1} K_{k-j} dW_j^P + c_bias_k

    Uses FFT for efficient O(T log T) computation.
    """
    T = dW_P.shape[1]
    K = kernel.shape[0]
    fft_len = T + K - 1

    dW_padded = jnp.pad(dW_P, ((0, 0), (0, fft_len - T), (0, 0)))
    kernel_padded = jnp.pad(kernel, ((0, fft_len - K), (0, 0), (0, 0)))

    dW_fft = jnp.fft.rfft(dW_padded, axis=1)
    kernel_fft = jnp.fft.rfft(kernel_padded, axis=0)

    u_fft = jnp.einsum("fij,bfj->bfi", kernel_fft, dW_fft)
    u = jnp.fft.irfft(u_fft, n=fft_len, axis=1)[:, :T, :]

    # Causal shift: u_k depends on dW_0..dW_{k-1} (not dW_k).
    u = jnp.pad(u[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

    if c_bias is not None:
        u = u + c_bias[None, :, :]
    return u


# =========================================================================
# 2. State-space tilt
# =========================================================================

def build_controller_affine_flows(
    rho: Array,
    U: Array,
    dW_P: Array,
) -> Tuple[Array, Array]:
    """
    Build affine flows for the diagonal state-space controller.

        s_{k+1} = diag(rho) s_k + U dW_k^P

    The recurrent state has dim ``state_dim = U.shape[0]`` which may
    differ from the driver dim (``state_expansion > 1`` widens the
    state without changing the input width).  We must therefore size
    ``F_ctrl`` from ``rho`` rather than from the driver.
    """
    B, T, _m_in = dW_P.shape
    state_dim = U.shape[0]
    F_ctrl = jnp.broadcast_to(rho[None, None, :], (B, T, state_dim))
    g_ctrl = jnp.einsum("ij,btj->bti", U, dW_P)
    return F_ctrl, g_ctrl


def controller_affine_combine(elem_a, elem_b):
    F_a, g_a = elem_a
    F_b, g_b = elem_b
    return (F_b * F_a, F_b * g_a + g_b)


def run_controller_scan(s0: Array, F_ctrl: Array, g_ctrl: Array) -> Array:
    """Sequential scan over the controller affine flows."""
    def step(s, flow):
        F, g = flow
        s_next = s * F + g
        return s_next, s_next

    _, s_seq = lax.scan(step, s0, (F_ctrl, g_ctrl))
    return jnp.concatenate([s0[None, :], s_seq], axis=0)


def run_controller_parallel_scan(
    s0: Array,
    F_ctrl: Array,
    g_ctrl: Array,
    parallel_steps: int,
) -> Array:
    """Chunked-parallel scan over the controller affine flows."""
    if parallel_steps == 1:
        return run_controller_scan(s0, F_ctrl, g_ctrl)

    T = F_ctrl.shape[0]
    m = s0.shape[0]

    remainder = T % parallel_steps
    if remainder == 0:
        core_F, core_g = F_ctrl, g_ctrl
        rem_F, rem_g = None, None
    else:
        core_F, core_g = F_ctrl[:-remainder], g_ctrl[:-remainder]
        rem_F, rem_g = F_ctrl[-remainder:], g_ctrl[-remainder:]

    n_chunks = core_F.shape[0] // parallel_steps
    core_F_chunks = core_F.reshape(n_chunks, parallel_steps, m)
    core_g_chunks = core_g.reshape(n_chunks, parallel_steps, m)

    def parallel_chunk_step(s, chunk):
        F_chunk, g_chunk = chunk
        prefix = lax.associative_scan(controller_affine_combine, (F_chunk, g_chunk))
        F_prefix, g_prefix = prefix
        ss = s[None, :] * F_prefix + g_prefix
        return ss[-1], ss

    _, ss_chunks = lax.scan(parallel_chunk_step, s0, (core_F_chunks, core_g_chunks))
    ss_core = ss_chunks.reshape(-1, m)
    ss_all = jnp.concatenate([s0[None, :], ss_core], axis=0)

    if rem_F is not None:
        s_last = ss_all[-1]

        def step(s, flow):
            F, g = flow
            s_next = s * F + g
            return s_next, s_next

        _, ss_rem = lax.scan(step, s_last, (rem_F, rem_g))
        ss_all = jnp.concatenate([ss_all, ss_rem], axis=0)

    return ss_all


# =========================================================================
# 3. RN weight and tilt helpers
# =========================================================================

def compute_rn_log_weight(u: Array, dW_P: Array, dt: Array) -> Array:
    """Discrete Radon-Nikodym log-weight.

        log(dQ/dP) = sum_k (u_k^T dW_k^P - 0.5 ||u_k||^2 dt)
    """
    cross = jnp.sum(u * dW_P, axis=-1)
    norm_sq = 0.5 * jnp.sum(u ** 2, axis=-1)
    log_rn = jnp.sum(cross - norm_sq * dt, axis=-1)
    return log_rn


def tilt_brownian_increments(dW_P: Array, u: Array, dt: Array) -> Array:
    """Tilt Brownian increments under the Girsanov measure change.

        dW^Q_k = dW^P_k + u_k dt
    """
    return dW_P + u * dt
