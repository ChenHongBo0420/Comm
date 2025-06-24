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
def qc_to_adj(H_qc: jnp.ndarray):
    """
    QC-LDPC (0/1) → (vn_adj, cn_adj)  
        vn_adj: (N, dv_max)   variable → check index，空位=-1  
        cn_adj: (M, dc_max)   check    → variable index，空位=-1
    """
    M, N   = H_qc.shape
    vn_idx = [jnp.where(H_qc[:, j])[0] for j in range(N)]
    cn_idx = [jnp.where(H_qc[i])[0]     for i in range(M)]
    dv_max = max(len(v) for v in vn_idx)
    dc_max = max(len(c) for c in cn_idx)

    pad = lambda arr, L: jnp.array(list(arr)+[-1]*(L-len(arr)), jnp.int32)
    vn_adj = jnp.stack([pad(v, dv_max) for v in vn_idx])   # (N,dv_max)
    cn_adj = jnp.stack([pad(c, dc_max) for c in cn_idx])   # (M,dc_max)
    return vn_adj, cn_adj
# ---------------------------------------------------------------------------#
# 4b - 稀疏 Neural-BP                                                        #
# ---------------------------------------------------------------------------#
class NeuralBP(nn.Module):
    """
    简化 Scaled-Min-Sum BP（无循环依赖状态，线程安全）

    Args
    ----
    vn_adj   : (N,dv_max) int32，variable-node 相邻 check 下标，空位=-1
    cn_adj   : (M,dc_max) int32，check-node  相邻 variable 下标，空位=-1
    n_iter   : BP 迭代次数
    learn_gamma : 若 True，则 γ 为可学习标量；否则 γ=1
    """
    vn_adj: jnp.ndarray
    cn_adj: jnp.ndarray
    n_iter: int = 5
    learn_gamma: bool = True

    def setup(self):
        if self.learn_gamma:
            self.gamma = self.param('gamma', nn.initializers.ones, ())

    # ------------------------------------------------------------------ #
    def __call__(self, llr0: jnp.ndarray) -> jnp.ndarray:     # llr0: (N,)
        γ         = self.gamma if self.learn_gamma else 1.0
        N, dv_max = self.vn_adj.shape
        M, dc_max = self.cn_adj.shape

        # 初始化 var → check 消息
        v2c = jnp.zeros((N, dv_max), llr0.dtype)              # (N,dv_max)

        # ---- 单次迭代 --------------------------------------------------- #
        def bp_step(v2c, _):
            # ------- check → var (scaled min-sum) -------- #
            msgs = v2c[self.cn_adj]                          # (M,dc_max,dv_max)
            # 当 dv_max==1 时 squeeze 掉最后一维，避免冗余
            if msgs.ndim == 3 and msgs.shape[-1] == 1:
                msgs = jnp.squeeze(msgs, -1)                 # → (M,dc_max)

            sign = jnp.prod(jnp.sign(msgs + 1e-12), axis=1)  # (M,)
            mag  = jnp.min(jnp.abs(msgs), axis=1)            # (M,)
            c2v  = γ * sign * mag                            # (M,)

            # ------- 聚合到 variable-nodes (scatter_add) --- #
            idx_flat   = self.cn_adj.flatten()               # (M*dc_max,)
            msg_flat   = jnp.repeat(c2v, dc_max)             # broadcast
            mask       = idx_flat >= 0
            idx_valid  = idx_flat[mask]
            msg_valid  = msg_flat[mask]

            v_sum = jnp.zeros(N, llr0.dtype).at[idx_valid].add(msg_valid)  # (N,)

            # ------- extrinsic & 更新 v2c ----------------- #
            v2c_new = (llr0 + v_sum)[:, None] - v2c          # broadcast减去旧消息
            return v2c_new, None

        # ---- scan 迭代 n_iter 次 --------------------------------------- #
        v2c_final, _ = lax.scan(bp_step, v2c, None, length=self.n_iter)

        # variable 节点最终 LLR
        llr_out = llr0 + jnp.sum(v2c_final, axis=1)
        return llr_out

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
