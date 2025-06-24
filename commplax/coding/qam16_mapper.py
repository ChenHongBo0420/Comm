# coding/qam16_mapper.py
import jax.numpy as jnp, jax

# 16-QAM Gray 表（已 √10 归一化）
CONST = jnp.array([
    -3-3j, -3-1j, -3+3j, -3+1j,
    -1-3j, -1-1j, -1+3j, -1+1j,
     3-3j,  3-1j,  3+3j,  3+1j,
     1-3j,  1-1j,  1+3j,  1+1j ], dtype=jnp.complex64) / jnp.sqrt(10.)

BIT_MAP = jnp.array([
   (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
   (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
   (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
   (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)
], dtype=jnp.float32)                              # shape (16,4)

@jax.jit
def bits_to_sym(bit4: jnp.ndarray) -> jnp.ndarray: # bit4 :(M,4)
    idx = jnp.dot(bit4, jnp.array([8,4,2,1], dtype=jnp.float32))
    return CONST[idx.astype(jnp.int32)]

def sym_to_bits(sym: jnp.ndarray) -> jnp.ndarray:  # 推理／BER 用
    dist = jnp.abs(sym[...,None] - CONST)          # (...,16)
    idx  = jnp.argmin(dist, axis=-1)
    return BIT_MAP[idx]
