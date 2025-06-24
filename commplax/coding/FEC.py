# coding/FEC.py
# ============================================================================
# 可微 QC-LDPC + 16QAM front-end  for commplax / GDBP pipeline
# ============================================================================
from __future__ import annotations
from typing import Dict, Tuple, Any

import jax, jax.numpy as jnp
import optax
from flax import linen as nn
import jax.lax  as lax
# ---------------------------------------------------------------------------#
# 0 - 引入可微 QC-LDPC 编码器 (STE)                                           #
# ---------------------------------------------------------------------------#
from commplax.coding.qc_ldpc_ste import qc_ldpc_encode, init_G_soft

try:
    from commplax.ldpc import qc_h_from_g    # 真正的 QC 展开
except ModuleNotFoundError:                  # --- 占位 (demo) ---
    def qc_h_from_g(G, Z):
        K, NK = G.shape
        return jnp.eye(K + NK, dtype=jnp.float32)

# ---------------------------------------------------------------------------#
# 1 - 16-QAM 常量 & Bit-map                                                  #
# ---------------------------------------------------------------------------#
_CONST = (jnp.array(
    [-3-3j,-3-1j,-3+3j,-3+1j,
     -1-3j,-1-1j,-1+3j,-1+1j,
      3-3j, 3-1j, 3+3j, 3+1j,
      1-3j, 1-1j, 1+3j, 1+1j], dtype=jnp.complex64) / jnp.sqrt(10.))

_BIT_MAP = jnp.array([
   (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
   (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
   (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
   (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)
], dtype=jnp.float32)
_BIT_W = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

# ---------------------------------------------------------------------------#
# 2 - Mapper / Demapper                                                      #
# ---------------------------------------------------------------------------#
def bits_to_sym(bits4: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.dot(bits4, jnp.array([8,4,2,1], dtype=jnp.float32))
    return _CONST[idx.astype(jnp.int32)]

def sym_to_bits(sym: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.argmin(jnp.abs(sym[...,None] - _CONST), axis=-1)
    return _BIT_MAP[idx]

# ---------------------------------------------------------------------------#
# 3 - Tx pipeline                                                            #
# ---------------------------------------------------------------------------#
def tx_pipeline(bits: jnp.ndarray, G_soft: jnp.ndarray,
                Π: jnp.ndarray | None = None) -> jnp.ndarray:
    cw   = qc_ldpc_encode(bits, G_soft)          # (N,)
    b4   = cw.reshape(-1,4)
    if Π is not None:
        b4 = b4 @ Π
    return bits_to_sym(b4)                       # (N/4,)

# ---------------------------------------------------------------------------#
# 4a - QC-matrix → 邻接表                                                    #
# ---------------------------------------------------------------------------#
def qc_to_adj(H: jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
    M,N = H.shape
    vn  = [jnp.where(H[:,j])[0] for j in range(N)]
    cn  = [jnp.where(H[i])[0]   for i in range(M)]
    dvM = max(len(v) for v in vn)
    dcM = max(len(c) for c in cn)
    pad = lambda li,L: jnp.pad(jnp.array(li, jnp.int32),
                               (0,L-len(li)), constant_values=-1)
    return jnp.stack([pad(v,dvM) for v in vn]), \
           jnp.stack([pad(c,dcM) for c in cn])

# ---------------------------------------------------------------------------#
# 4b - 稀疏 Neural-BP                                                        #
# ---------------------------------------------------------------------------#
class NeuralBP(nn.Module):
    vn_adj: jnp.ndarray      # (N,dv)
    cn_adj: jnp.ndarray      # (M,dc)
    n_iter: int = 5
    learn_gamma: bool = True
    def setup(self):
        if self.learn_gamma:
            self.gamma = self.param('gamma', nn.initializers.ones, ())
    def __call__(self, llr0: jnp.ndarray):       # (N,)
        γ    = self.gamma if self.learn_gamma else 1.0
        v2c  = jnp.zeros_like(self.vn_adj, dtype=llr0.dtype)  # (N,dv)

        def step(v2c,_):
            # ---- check → var ----
            msg   = v2c[self.cn_adj]                           # (M,dc)
            mask  = (self.cn_adj < 0)
            msg   = jnp.where(mask, 0.0, msg)                  # padding 0
            sgn   = jnp.prod(jnp.sign(msg+1e-12), axis=1, keepdims=True)
            mag   = jnp.min(jnp.abs(jnp.where(mask, 1e9, msg)), axis=1,
                            keepdims=True)
            c2v   = γ * sgn * mag
            c2v   = jnp.broadcast_to(c2v, self.cn_adj.shape)
            c2v   = jnp.where(mask, 0.0, c2v)

            # scatter-add 到 variable
            v_acc = jnp.zeros_like(v2c)
            v_acc = v_acc.at[self.cn_adj].add(c2v)

            v2c_new = (llr0[:,None] + v_acc) - v2c
            v2c_new = jnp.where(self.vn_adj<0, 0.0, v2c_new)
            return v2c_new, None

        v2c,*_ = lax.scan(step, v2c, None, length=self.n_iter)
        return llr0 + jnp.sum(v2c, axis=1)

# ---------------------------------------------------------------------------#
# 5 - bit-BCE loss                                                          #
# ---------------------------------------------------------------------------#
@jax.jit
def bit_bce_loss(pred: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    logits = -jnp.square(jnp.abs(pred[...,None] - _CONST))
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    p1     = jnp.exp(logp) @ _BIT_MAP
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.abs(ref[...,None]-_CONST), axis=-1)
    bits_t = _BIT_MAP[idx]
    bce    = -(bits_t*jnp.log(p1+1e-12) + (1.-bits_t)*jnp.log(p0+1e-12))
    return jnp.mean(bce * _BIT_W)

# ---------------------------------------------------------------------------#
# 6 - 三阶段 Optax optimizer                                                #
# ---------------------------------------------------------------------------#
def build_optimizers(pytree: Dict[str,Any],
                     lr_dsp=3e-5, lr_fec=1e-4, lr_jnt=3e-5):
    """pytree 必须包含键 dsp / G / Π / bp"""
    is_dsp = lambda path,_: path and path[0]=='dsp'
    opt1 = optax.multi_transform({'dsp': optax.adam(lr_dsp)},
                                 param_labels=jax.tree_util.Partial(is_dsp))
    opt2 = optax.multi_transform({'fec': optax.adam(lr_fec)},
                                 param_labels=lambda p,_: 'fec')
    opt3 = optax.adam(lr_jnt)
    return opt1, opt2, opt3

# ---------------------------------------------------------------------------#
# 7 - 导出硬表                                                               #
# ---------------------------------------------------------------------------#
def export_ldpc_tables(G_soft: jnp.ndarray, Z:int=512,
                       out_csv:str="new_qc_ldpc.csv") -> jnp.ndarray:
    import numpy as np, csv
    G_hard = (jax.device_get(G_soft) > 0.5).astype(np.int8)
    np.savetxt(out_csv, G_hard, fmt='%d', delimiter=',')
    return qc_h_from_g(G_hard, Z)            # parity 矩阵（float32）
