# ---------- qam16_soft.py ----------
import jax.numpy as jnp

# 16-QAM 星座（√10 归一）
CONST = jnp.array([
    -3-3j,-3-1j,-3+3j,-3+1j,
    -1-3j,-1-1j,-1+3j,-1+1j,
     3-3j, 3-1j, 3+3j, 3+1j,
     1-3j, 1-1j, 1+3j, 1+1j
], dtype=jnp.complex64) / jnp.sqrt(10.)

# 16×4 bit 查表（Gray）
BITS = jnp.array([
 [0,0,0,0],[0,0,0,1],[0,0,1,1],[0,0,1,0],
 [0,1,0,0],[0,1,0,1],[0,1,1,1],[0,1,1,0],
 [1,1,0,0],[1,1,0,1],[1,1,1,1],[1,1,1,0],
 [1,0,0,0],[1,0,0,1],[1,0,1,1],[1,0,1,0],
], dtype=jnp.int8)                      # (16,4)

def sym2bit(symbols: jnp.ndarray) -> jnp.ndarray:
    """complex (N,) → int8 (N,4)"""
    idx = jnp.argmin(jnp.abs(symbols[:,None]-CONST[None,:])**2, axis=1)
    return BITS[idx]

def llr_maxlog(y: jnp.ndarray, noise_var: float) -> jnp.ndarray:
    """
    y:(N,) complex → LLR:(N,4) (正 ⇒ bit=0 概率大)
    Max-log-MAP 距离近似
    """
    d2 = jnp.abs(y[:,None]-CONST[None,:])**2 / noise_var        # (N,16)
    def _one(b):
        m0 = jnp.min(jnp.where(BITS[:,b]==0, d2, jnp.inf), axis=1)
        m1 = jnp.min(jnp.where(BITS[:,b]==1, d2, jnp.inf), axis=1)
        return m1 - m0
    return jnp.stack([_one(i) for i in range(4)], axis=1)
