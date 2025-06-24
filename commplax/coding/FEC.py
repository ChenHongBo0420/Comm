# coding/FEC.py
# ============================================================================
# 可微 QC-LDPC + 16QAM front-end  for commplax / GDBP pipeline
# -  qc_ldpc_encode (STE) & init_G_soft
# -  16-QAM Gray mapper / demapper
# -  Neural-BP 解码器（可学习 γ）
# -  bit-BCE 损失
# -  三阶段 Optax optimizer 生成
# 作者:  your-name   2025-06-24
# ============================================================================

from __future__ import annotations
from typing import Dict, Tuple, Any

import jax, jax.numpy as jnp
import optax
from flax import linen as nn

# ---------------------------------------------------------------------------#
# 0 -  引入可微 LDPC 编码器 (STE)                                             #
# ---------------------------------------------------------------------------#
from commplax.coding.qc_ldpc_ste import qc_ldpc_encode, init_G_soft

# 若你有 commplax.ldpc 的 QC 工具，可直接 import；否则占位
try:
    from commplax.ldpc import qc_h_from_g         # (G_hard, Z) -> H
except ModuleNotFoundError:                       # 占位：identity (demo)
    def qc_h_from_g(G, Z):          # G:(K,N-K)
        """⚠️ 仅占位 - 请换成真实 QC 展开"""
        K, NK = G.shape
        N = K + NK
        return jnp.eye(N, dtype=jnp.float32)      # 不做校验

# ---------------------------------------------------------------------------#
# 1 -  16-QAM 常量 & Bit-map                                                  #
# ---------------------------------------------------------------------------#
_CONST = (jnp.array(
    [-3-3j,-3-1j,-3+3j,-3+1j,
     -1-3j,-1-1j,-1+3j,-1+1j,
      3-3j, 3-1j, 3+3j, 3+1j,
      1-3j, 1-1j, 1+3j, 1+1j], dtype=jnp.complex64)
          / jnp.sqrt(10.))            # Gray-mapped & √10 归一

_BIT_MAP = jnp.array([
   (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
   (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
   (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
   (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)
], dtype=jnp.float32)                                 # (16,4)

# bit 权重  — 可调
_BIT_W = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

# ---------------------------------------------------------------------------#
# 2 -  Mapper / Demapper                                                     #
# ---------------------------------------------------------------------------#
def bits_to_sym(bits4: jnp.ndarray) -> jnp.ndarray:        # (M,4)
    """4-bit → 16-QAM complex"""
    idx = jnp.dot(bits4, jnp.array([8,4,2,1], dtype=jnp.float32))
    return _CONST[idx.astype(jnp.int32)]

def sym_to_bits(sym: jnp.ndarray) -> jnp.ndarray:          # (...,)→(...,4)
    dist = jnp.abs(sym[..., None] - _CONST)
    idx  = jnp.argmin(dist, axis=-1)
    return _BIT_MAP[idx]

# ---------------------------------------------------------------------------#
# 3 -  Differentiable Tx pipeline                                            #
# ---------------------------------------------------------------------------#
def tx_pipeline(bits: jnp.ndarray,           # (K,)
                G_soft: jnp.ndarray,         # (K, N-K)
                Π: jnp.ndarray | None = None # (4,4) 可学置换
                ) -> jnp.ndarray:
    """bit-block → LDPC encode → symbol stream"""
    cw = qc_ldpc_encode(bits, G_soft)        # (N,)
    assert cw.size % 4 == 0, "Codeword length must be 4×integer"
    bit4 = cw.reshape(-1, 4)                 # (N/4,4)
    if Π is not None:
        bit4 = jnp.matmul(bit4, Π)           # 置换 / P-allocation
    return bits_to_sym(bit4)                 # (N/4,) complex

# ---------------------------------------------------------------------------#
# 4 -  Neural-BP Decoder (简版 Scaled Min-Sum)                                #
# ---------------------------------------------------------------------------#
class NeuralBP(nn.Module):
    H:  jnp.ndarray        # (M,N) parity
    n_iter: int = 5
    learn_gamma: bool = True
    def setup(self):
        if self.learn_gamma:
            self.gamma = self.param('gamma', nn.initializers.ones, ())
    def __call__(self, llr: jnp.ndarray) -> jnp.ndarray:   # (N,)
        γ = self.gamma if self.learn_gamma else 1.0
        v2c = jnp.tile(llr, (self.H.shape[0],1))           # (M,N)
        for _ in range(self.n_iter):
            # check-to-var: scaled-min-sum
            c2v_sign = jnp.sign(self.H @ jnp.sign(v2c.T))
            c2v_mag  = jnp.min(jnp.where(self.H, jnp.abs(v2c), 1e9), axis=1, keepdims=True)
            c2v      = γ * c2v_sign * c2v_mag
            # var-to-check
            v2c = llr + (self.H.T @ c2v)
        return llr + (self.H.T @ c2v)        # extrinsic soft bit LLR

# ---------------------------------------------------------------------------#
# 5 -  Bit-BCE 损失                                                          #
# ---------------------------------------------------------------------------#
@jax.jit
def bit_bce_loss(pred_sym: jnp.ndarray,
                 true_sym: jnp.ndarray) -> jnp.ndarray:
    """weighted bit-level BCE on symbols"""
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - _CONST))      # (...,16)
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)
    p1     = probs @ _BIT_MAP
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(true_sym[...,None]-_CONST)), axis=-1)
    bits_t = _BIT_MAP[idx]
    bce    = -(bits_t * jnp.log(p1+1e-12) + (1.-bits_t)*jnp.log(p0+1e-12))
    return (bce * _BIT_W).mean()

# ---------------------------------------------------------------------------#
# 6 -  三阶段 optimizer 生成                                                 #
# ---------------------------------------------------------------------------#
def build_optimizers(param_tree: Dict[str,Any],
                     lr1=1e-4, lr2=1e-4, lr3=1e-5):
    """
    param_tree keys 必须包含:  dsp / G / Π / bp
    """
    # PyTree 布尔掩码
    mask_all = jax.tree_util.tree_map(lambda _: True, param_tree)
    mask_dsp = jax.tree_util.tree_map(lambda _: False, param_tree)
    mask_dsp['dsp'] = True

    mask_GΠbp = jax.tree_util.tree_map(lambda _: True, param_tree)
    mask_GΠbp['dsp'] = False

    opt1 = optax.masked(optax.adam(lr1), mask_dsp)
    opt2 = optax.masked(optax.adam(lr2), mask_GΠbp)
    opt3 = optax.adam(lr3)                     # 全部参数

    return opt1, opt2, opt3

# ---------------------------------------------------------------------------#
# 7 -  辅助:  从已训练 G_soft 导出硬码表 & H                                 #
# ---------------------------------------------------------------------------#
def export_ldpc_tables(G_soft: jnp.ndarray,
                       Z: int = 512,
                       path: str = "new_qc_ldpc.csv") -> jnp.ndarray:
    """
    round(G_soft) → 保存 csv → 生成 H
    """
    import numpy as np
    G_hard = (jax.device_get(G_soft) > 0.5).astype(np.int8)
    np.savetxt(path, G_hard, fmt='%d', delimiter=',')
    H = qc_h_from_g(G_hard, Z)                 # parity matrix
    return jnp.asarray(H, dtype=jnp.float32)
