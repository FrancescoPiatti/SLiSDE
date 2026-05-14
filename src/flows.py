# v3/src/flows.py
"""Affine flow construction, composition, scan and offset injection.

Each base SDE layer is the affine recurrence
    Z_{k+1} = F_k Z_k + g_k,
with
    F_k = exp(M_k) or  I + M_k,   M_k = A_k dt + Σ_j C_k^j dW^j_k,
    g_k = b_k dt + D_k dW_k.

The affine combine
    (F_b, g_b) ∘ (F_a, g_a) = (F_b F_a, F_b g_a + g_b)
is associative, so a sequence of ``(F_k, g_k)`` can be scanned in parallel.

Static vs. time-dependent coefficients
--------------------------------------
The ``build_affine_flows_*`` builders accept the static parameters
``A, b, C, dense_D`` plus *optional* decoder arrays and a time embedding
``h_t``. If any decoder is non-None we apply the corresponding
time-dependent correction on top of the static coefficient; with every
decoder set to ``None`` the function reduces to the static SDE used in v2.

Matrix-type dispatch is done at the SDE layer (see ``layers/base.py``).
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import expm

Array = jnp.ndarray


# =========================================================================
# 1. Affine flow builders (per matrix type, time-dependent when decoders
#    are supplied; degenerate to the static case when they are ``None``).
# =========================================================================

def build_affine_flows_diag(
    *,
    A: Array,
    b: Array,
    dt: Array,
    dW: Array,
    noise_type: str,
    approx_exp: bool,
    C: Optional[Array] = None,
    dense_D: Optional[Array] = None,
    h_t: Optional[Array] = None,
    A_decoder: Optional[Array] = None,
    b_decoder: Optional[Array] = None,
    C_decoder: Optional[Array] = None,
    D_decoder: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Build ``(F, g)`` for the diagonal-matrix case.

    Without decoders this is the static SDE used in v2. Pass ``h_t`` plus
    one or more ``*_decoder`` arrays to add time-dependent corrections.
    """
    B_size, T_size = dW.shape[:2]
    d = A.shape[0]

    # ------ A and the drift contribution to M ----------------------------
    if A_decoder is not None and h_t is not None:
        # A_decoder: (d, H);  dA_t: (T, d)
        dA_t = jnp.einsum("th,dh->td", h_t, A_decoder)
        A_t = A[None, :] + dA_t                                # (T, d)
        M_drift = dt * A_t[None, :, :]                         # (1, T, d)
    else:
        M_drift = dt * A[None, None, :]                        # (1, 1, d)

    # ------ Multiplicative diffusion contribution to M -------------------
    if noise_type in ("multiplicative", "general"):
        if C_decoder is not None and h_t is not None:
            dC_t = jnp.einsum("th,mdh->tmd", h_t, C_decoder)   # (T, m, d)
            C_t = C[None, :, :] + dC_t                          # (T, m, d)
            M_noise = jnp.einsum("btm,tmd->btd", dW, C_t)
        else:
            M_noise = jnp.einsum("btm,md->btd", dW, C)
        M = M_drift + M_noise
    else:
        M = jnp.broadcast_to(M_drift, (B_size, T_size, d))

    F = 1.0 + M if approx_exp else jnp.exp(M)

    # ------ Bias and the additive diffusion contribution to g ------------
    if b_decoder is not None and h_t is not None:
        db_t = jnp.einsum("th,dh->td", h_t, b_decoder)         # (T, d)
        b_t = b[None, :] + db_t                                # (T, d)
        g = dt * b_t[None, :, :]                               # (1, T, d)
    else:
        g = (dt * b)[None, None, :]                            # (1, 1, d)
    g = jnp.broadcast_to(g, (B_size, T_size, d))

    if noise_type in ("additive", "general"):
        if D_decoder is not None and h_t is not None:
            # Per-noise-channel amplitude: scale_t: (T, m)
            scale_t = 1.0 + jnp.einsum("th,hm->tm", h_t, D_decoder)
            g = g + jnp.einsum("btm,tm,md->btd", dW, scale_t, dense_D)
        else:
            g = g + jnp.einsum("btm,md->btd", dW, dense_D)

    return F, g


def build_affine_flows_blockdiag(
    *,
    A: Array,
    b: Array,
    dt: Array,
    dW: Array,
    noise_type: str,
    approx_exp: bool,
    C: Optional[Array] = None,
    dense_D: Optional[Array] = None,
    h_t: Optional[Array] = None,
    A_decoder: Optional[Array] = None,
    b_decoder: Optional[Array] = None,
    C_decoder: Optional[Array] = None,
    D_decoder: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Build ``(F, g)`` for the block-diagonal-matrix case."""
    nb, bs, _ = A.shape
    d = nb * bs
    B_size, T_size = dW.shape[:2]

    if A_decoder is not None and h_t is not None:
        # A_decoder: (nb, bs, bs, H)
        dA_t = jnp.einsum("th,nijh->tnij", h_t, A_decoder)     # (T, nb, bs, bs)
        A_t = A[None, :, :, :] + dA_t
        M_drift = dt * A_t[None, :, :, :, :]                   # (1, T, nb, bs, bs)
    else:
        M_drift = dt * A[None, None, :, :, :]

    if noise_type in ("multiplicative", "general"):
        if C_decoder is not None and h_t is not None:
            dC_t = jnp.einsum("th,mnijh->tmnij", h_t, C_decoder)
            C_t = C[None, :, :, :, :] + dC_t
            M_noise = jnp.einsum("btm,tmnij->btnij", dW, C_t)
        else:
            M_noise = jnp.einsum("btm,mnij->btnij", dW, C)
        M = M_drift + M_noise
    else:
        M = jnp.broadcast_to(M_drift, (B_size, T_size, nb, bs, bs))

    if approx_exp:
        eye = jnp.eye(bs, dtype=M.dtype)[None, None, None, :, :]
        F = eye + M
    else:
        F = jax.vmap(jax.vmap(jax.vmap(expm)))(M)

    # Bias / additive diffusion are flat, then reshaped to block layout.
    if b_decoder is not None and h_t is not None:
        db_t = jnp.einsum("th,dh->td", h_t, b_decoder)
        b_t = b[None, :] + db_t                                # (T, d)
        g_flat = dt * b_t[None, :, :]
    else:
        g_flat = (dt * b)[None, None, :]
    g_flat = jnp.broadcast_to(g_flat, (B_size, T_size, d))

    if noise_type in ("additive", "general"):
        if D_decoder is not None and h_t is not None:
            scale_t = 1.0 + jnp.einsum("th,hm->tm", h_t, D_decoder)
            g_flat = g_flat + jnp.einsum("btm,tm,md->btd", dW, scale_t, dense_D)
        else:
            g_flat = g_flat + jnp.einsum("btm,md->btd", dW, dense_D)

    return F, g_flat.reshape(B_size, T_size, nb, bs)


def build_affine_flows_dense(
    *,
    A: Array,
    b: Array,
    dt: Array,
    dW: Array,
    noise_type: str,
    approx_exp: bool,
    C: Optional[Array] = None,
    dense_D: Optional[Array] = None,
    h_t: Optional[Array] = None,
    A_decoder: Optional[Array] = None,
    b_decoder: Optional[Array] = None,
    C_decoder: Optional[Array] = None,
    D_decoder: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Build ``(F, g)`` for the dense-matrix case."""
    d = A.shape[0]
    B_size, T_size = dW.shape[:2]

    if A_decoder is not None and h_t is not None:
        dA_t = jnp.einsum("th,ijh->tij", h_t, A_decoder)        # (T, d, d)
        A_t = A[None, :, :] + dA_t
        M_drift = dt * A_t[None, :, :, :]                       # (1, T, d, d)
    else:
        M_drift = dt * A[None, None, :, :]

    if noise_type in ("multiplicative", "general"):
        if C_decoder is not None and h_t is not None:
            dC_t = jnp.einsum("th,mijh->tmij", h_t, C_decoder)
            C_t = C[None, :, :, :] + dC_t
            M_noise = jnp.einsum("btm,tmij->btij", dW, C_t)
        else:
            M_noise = jnp.einsum("btm,mij->btij", dW, C)
        M = M_drift + M_noise
    else:
        M = jnp.broadcast_to(M_drift, (B_size, T_size, d, d))

    if approx_exp:
        eye = jnp.eye(d, dtype=M.dtype)[None, None, :, :]
        F = eye + M
    else:
        F = jax.vmap(jax.vmap(expm))(M)

    if b_decoder is not None and h_t is not None:
        db_t = jnp.einsum("th,dh->td", h_t, b_decoder)
        b_t = b[None, :] + db_t
        g = dt * b_t[None, :, :]
    else:
        g = (dt * b)[None, None, :]
    g = jnp.broadcast_to(g, (B_size, T_size, d))

    if noise_type in ("additive", "general"):
        if D_decoder is not None and h_t is not None:
            scale_t = 1.0 + jnp.einsum("th,hm->tm", h_t, D_decoder)
            g = g + jnp.einsum("btm,tm,md->btd", dW, scale_t, dense_D)
        else:
            g = g + jnp.einsum("btm,md->btd", dW, dense_D)

    return F, g


# =========================================================================
# 2. Offset injection (used by gated in-flow layers)
# =========================================================================

def inject_offset_diag(flows, offset):
    F, g = flows
    return F, g + offset


def inject_offset_blockdiag(flows, offset):
    F, g = flows
    _, _, nb, bs = g.shape
    return F, g + offset.reshape(offset.shape[0], offset.shape[1], nb, bs)


def inject_offset_dense(flows, offset):
    F, g = flows
    return F, g + offset


# =========================================================================
# 3. Affine combine rules (associative scan operators)
# =========================================================================

def affine_combine_diag(elem_a, elem_b):
    F_a, g_a = elem_a
    F_b, g_b = elem_b
    return F_b * F_a, F_b * g_a + g_b


def affine_combine_blockdiag(elem_a, elem_b):
    F_a, g_a = elem_a
    F_b, g_b = elem_b
    return jnp.matmul(F_b, F_a), jnp.einsum("...nij,...nj->...ni", F_b, g_a) + g_b


def affine_combine_dense(elem_a, elem_b):
    F_a, g_a = elem_a
    F_b, g_b = elem_b
    return (
        jnp.einsum("...ij,...jk->...ik", F_b, F_a),
        jnp.einsum("...ij,...j->...i", F_b, g_a) + g_b,
    )


# =========================================================================
# 4. Sequential step functions (for jax.lax.scan)
# =========================================================================

def affine_step_diag(z, flow):
    F, g = flow
    z_next = z * F + g
    return z_next, z_next


def affine_step_blockdiag(z, flow):
    F, g = flow
    nb, bs, _ = F.shape
    z_blk = z.reshape(nb, bs)
    z_next = (jnp.einsum("nij,nj->ni", F, z_blk) + g).reshape(-1)
    return z_next, z_next


def affine_step_dense(z, flow):
    F, g = flow
    z_next = jnp.einsum("ij,j->i", F, z) + g
    return z_next, z_next


# =========================================================================
# 5. Parallel step functions (chunked associative scan)
# =========================================================================

def parallel_affine_step_diag(z0, flow_chunk):
    F_prefix, g_prefix = lax.associative_scan(affine_combine_diag, flow_chunk)
    zs = z0[None, :] * F_prefix + g_prefix
    return zs[-1], zs


def parallel_affine_step_blockdiag(z0, flow_chunk):
    F_prefix, g_prefix = lax.associative_scan(affine_combine_blockdiag, flow_chunk)
    nb, bs = F_prefix.shape[1], F_prefix.shape[2]
    z_blk = z0.reshape(nb, bs)
    zs_blk = jnp.einsum("knij,nj->kni", F_prefix, z_blk) + g_prefix
    zs = zs_blk.reshape(F_prefix.shape[0], -1)
    return zs[-1], zs


def parallel_affine_step_dense(z0, flow_chunk):
    F_prefix, g_prefix = lax.associative_scan(affine_combine_dense, flow_chunk)
    zs = jnp.einsum("kij,j->ki", F_prefix, z0) + g_prefix
    return zs[-1], zs
