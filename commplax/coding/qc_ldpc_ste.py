# coding/qc_ldpc_ste.py
"""
Differentiable QC-LDPC encoder with Straight-Through Estimator (STE)

* Author:  C HB
* Date  :  2025-06-25
"""

import jax, jax.numpy as jnp
from functools import partial
from typing import Tuple

# ---------- helper ---------------------------------------------------------- #
def init_G_soft(key: jax.Array,
                mask: jax.Array,
                std: float = 0.01) -> jax.Array:
    """
    Initialise soft generator matrix G_soft.

    Args:
        key  : PRNGKey
        mask : (K, N-K) 0/1 array. 1 表示该位置允许为 1（源于 protograph）。
        std  : random noise std 用于微扰 0.5

    Returns:
        G_soft : float32, same shape as mask, entries ∈ (0,1)
    """
    noise = std * jax.random.normal(key, mask.shape)
    return (0.5 + noise) * mask.astype(jnp.float32)


# ---------- STE encoder ----------------------------------------------------- #
@jax.custom_vjp
def qc_ldpc_encode(bits: jax.Array,       # (K,)
                   G_soft: jax.Array      # (K, N-K)
) -> jax.Array:
    """
    Forward:  round(G_soft) 作为硬 0/1 → 经典 QC-LDPC systematic encode.
    Backward:  ∂cw/∂G_soft = upstream_grad  (STE – 直通)

    Returns:
        cw : (N,)  codeword, float32 ∈ {0,1}
    """
    P = jnp.round(G_soft)                               # STE 硬化
    parity = jnp.mod(jnp.matmul(bits, P), 2.0)          # xor  (K @ (K, N-K) → N-K)
    return jnp.concatenate([bits, parity], axis=0)      # (N,)

# ---- VJP rules ---- #
def _fwd(bits, G_soft):
    cw = qc_ldpc_encode(bits, G_soft)
    return cw, (bits.shape, G_soft)                     # save shape info

def _bwd(res, g_cw):
    bits_shape, G_soft = res
    # ∂cw/∂bits = [I_K | P]，我们只关心传到 G_soft 的梯度
    grad_bits   = jnp.zeros(bits_shape, dtype=g_cw.dtype)
    grad_Gsoft  = g_cw[bits_shape[0]:]                  # upstream grad on parity part
    return grad_bits, grad_Gsoft

qc_ldpc_encode.defvjp(_fwd, _bwd)
