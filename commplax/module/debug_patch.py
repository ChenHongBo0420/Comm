# ── debug_patch.py ──────────────────────────────────────────────
"""
把 conv1d_fft / dconv_pair 换成带 nan-guard + 打印的 debug 版本，
并附带若干辅助检查函数。import 本文件即可自动 monkey-patch。
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from functools import partial
import numpy as np

# ---------- 0. 依赖原 core -------------------------------------------------
from commplax.module import core
SigTime, Signal = core.SigTime, core.Signal
delta = core.delta
wpartial = core.wpartial
# ---------------------------------------------------------------------------


# ---------- 1. 安全-FFT 卷积 (monkey-patch) ---------------------------------
def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def conv1d_fft_debug(scope,
                     signal: Signal,
                     *,
                     taps: int = 261,
                     seglen: int | None = None,
                     kernel_init = delta,
                     debug: bool = True,
                     max_fftlen: int = 2**18):      # 256k 点再大很容易溢 mem
    """
    完全复写 core.conv1d_fft，**多做 4 件事**
      ① 自动裁剪过大 FFT 长度
      ② 运行期 (not initializing) 检 nan/inf 并报首地址
      ③ debug 时打印 slice / fftlen / out_len
      ④ 对输出再做一次 finite 过滤（把异常置 0，防梯度爆）
    """
    x, t_in = signal
    h = scope.param('kernel', kernel_init, (taps,), jnp.complex64)

    rtap   = (taps - 1) // 2
    N_out  = x.shape[0] - 2 * rtap
    fftlen = seglen or _next_pow2(x.shape[0] + taps - 1)
    fftlen = int(min(fftlen, max_fftlen))

    Xk = jnp.fft.fft(x, fftlen, axis=0)
    Hk = jnp.fft.fft(h, fftlen);  Hk = Hk if x.ndim==1 else Hk[:,None]
    y_full = jnp.fft.ifft(Xk * Hk, fftlen, axis=0)
    y_val  = y_full[rtap : rtap + N_out]

    # ---- nan / inf 守卫 ---------------------------------------------------
    if debug and (not scope.is_initializing()):
        bad = ~jnp.isfinite(y_val)
        if bad.any():
            idx = int(jnp.where(bad)[0][0])
            print(f"⚠️ conv1d_fft nan/inf  @ out[{idx}]  value={y_val[idx]}")
            raise FloatingPointError("conv1d_fft produced non-finite value")

    y_val = jnp.where(jnp.isfinite(y_val), y_val, 0.0)     # clamp

    if debug and scope.is_initializing():
        print(f"[conv1d_fft] N={x.shape[0]} taps={taps} "
              f"rtap={rtap}  fftlen={fftlen}  out_len={N_out}")

    t_out = SigTime(t_in.start + rtap, t_in.stop - rtap, t_in.sps)
    return Signal(y_val, t_out)

# -> 打开猴子补丁
core.conv1d_fft = conv1d_fft_debug
# ---------------------------------------------------------------------------


# ---------- 2. 双极化 wrapper (带 debug) ------------------------------------
def dconv_pair_debug(scope,
                     sig: Signal,
                     *,
                     taps: int,
                     kinit = delta,
                     debug: bool = True):
    """
    两极化分开跑 conv1d_fft_debug，再 concat，保持原 API。
    """
    x, t = sig
    outs = []
    for p in range(x.shape[1]):                            # (N,2)
        y, tp = scope.child(
            partial(conv1d_fft_debug,
                    taps=taps,
                    kernel_init=kinit,
                    debug=debug),
            name=f'Pol{p}'
        )(Signal(x[:, p], t))
        outs.append(y[:, None])
    return Signal(jnp.concatenate(outs, axis=1), tp)

core.dconv_pair = dconv_pair_debug   # monkey-patch
# ---------------------------------------------------------------------------


# ---------- 3. 对等性检查工具 ----------------------------------------------
def check_pair_equivalent(N:int=2048, taps:int=271, key:int=0):
    """
    δ-kernel 时 dconv_pair 应当仅作中心裁切 —— 用来 sanity-check.
    """
    k = jax.random.PRNGKey(key)
    x = (jax.random.normal(k, (N,2)) +
         1j*jax.random.normal(k, (N,2))).astype(jnp.complex64)

    scope = core.FakeScope(delta(None, (taps,)))
    y,_ = dconv_pair_debug(scope,
                           Signal(x, SigTime(0,0,2)),
                           taps=taps,
                           kinit=delta,
                           debug=False)

    assert jnp.allclose(y.val, x[taps//2:-taps//2]), \
           "dconv_pair mismatch with δ-kernel!"
    print("✓ dconv_pair equivalence passed.")

# ---------------------------------------------------------------------------


# ---------- 4. 训练时能量打印辅助 -------------------------------------------
def print_pol_energy(step:int, z: jnp.ndarray, x: jnp.ndarray):
    """
    z,x: (L, C) complex
    """
    rms_z = jnp.sqrt(jnp.mean(jnp.abs(z)**2, axis=0))
    rms_x = jnp.sqrt(jnp.mean(jnp.abs(x)**2, axis=0))
    print(f"[ENERGY] step={step:4d}  ‖z‖={np.asarray(rms_z)}  "
          f"‖x‖={np.asarray(rms_x)}")

# 如果你已经在别处定义过同名函数，就覆盖它，方便外部直接 import
import sys
sys.modules[__name__+'.print_pol_energy']=print_pol_energy
# ---------------------------------------------------------------------------


# ---------- 5. 自动在 import 时跑一次单极化自测 ------------------------------
if __name__ == "__main__":
    # 只手动运行 debug_patch.py 时执行；被 import 时不会影响速度
    N, taps = 4097, 271
    δ = (jnp.arange(N) == 0).astype(jnp.complex64)
    scope = core.FakeScope(delta(None,(taps,)))
    y,_ = conv1d_fft_debug(scope, Signal(δ, SigTime(0,0,1)), taps=taps, debug=True)
    ref = jnp.convolve(δ, scope.h[::-1], 'valid')
    assert jnp.max(jnp.abs(y.val - ref)) < 1e-6
    check_pair_equivalent()
    print("✓ all debug-patch self-tests passed")
# ─────────────────────────────────────────────────────────────────
