# ── debug_patch.py ─────────────────────────────────────────────────────────
"""
给 conv1d_fft / dconv_pair 打 debug-guard：
  • 初始化阶段仅打印一次 kernel / fftlen
  • 训练阶段一旦发现 nan/inf 立即抛错
把本文件放在工作目录，训练脚本最先 `import debug_patch`
"""

import types, functools, numpy as np, jax.numpy as jnp
from flax.core import Scope
from commplax.module import core                      # 原库

# ---------- helpers --------------------------------------------------------
def _is_init(scope: Scope) -> bool:
    """兼容不同 Flax 版本；无此属性时返回 False"""
    f = getattr(scope, 'is_initializing', None)
    return f() if callable(f) else False


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ---------- monkey-patch conv1d_fft ----------------------------------------
def conv1d_fft_debug(scope: Scope,
                     signal: core.Signal,
                     *,
                     taps: int = 261,
                     seglen: int | None = None,
                     kernel_init = core.delta,
                     debug: bool = True,
                     max_fftlen: int | None = None):

    x, t_in = signal
    h = scope.param('kernel', kernel_init, (taps,), jnp.complex64)

    rtap   = (taps - 1) // 2
    Nout   = x.shape[0] - 2 * rtap
    fftlen = seglen or _next_pow2(x.shape[0] + taps - 1)
    if max_fftlen is not None:
        fftlen = min(fftlen, max_fftlen)            # ← 可选上限

    # --- FFT 卷积 -----------------------------------------------------------
    Xk = jnp.fft.fft(x,      fftlen, axis=0)
    Hk = jnp.fft.fft(h,      fftlen)
    if x.ndim == 2:
        Hk = Hk[:, None]
    y_full = jnp.fft.ifft(Xk * Hk, fftlen, axis=0)
    y_val  = y_full[rtap: rtap + Nout]

    t_out  = core.SigTime(t_in.start + rtap,
                          t_in.stop  - rtap,
                          t_in.sps)

    # --- debug print --------------------------------------------------------
    if debug and _is_init(scope):
        print(f"[conv1d_fft] N={x.shape[0]} taps={taps}"
              f" rtap={rtap} fftlen={fftlen}  out_len={Nout}")

    # ---- nan / inf 守卫 ----------------------------------------------------
    if debug and (not _is_init(scope)):
        if not jnp.isfinite(y_val).all():
            bad = jnp.where(~jnp.isfinite(y_val))[0][0]
            raise FloatingPointError(
                f"⚠️ conv1d_fft nan/inf  @ out[{int(bad)}] value={y_val[bad]}")

    return core.Signal(y_val, t_out)


# ---------- monkey-patch dconv_pair ----------------------------------------
def dconv_pair_debug(scope: Scope,
                     sig: core.Signal,
                     *,
                     taps: int,
                     kinit,
                     debug: bool = True):

    x, t = sig
    outs = []
    for p in range(x.shape[1]):
        y, tp = scope.child(
            functools.partial(conv1d_fft_debug,
                              taps=taps,
                              kernel_init=kinit,
                              debug=debug),
            name=f'Pol{p}'
        )(core.Signal(x[:, p], t))
        outs.append(y[:, None])
    return core.Signal(jnp.concatenate(outs, axis=1), tp)


# ---------- 注册到 commplax.core ------------------------------------------
core.conv1d_fft = conv1d_fft_debug           # 单极化
core.dconv_pair = dconv_pair_debug           # 双极化
print("✓ debug_patch loaded — conv1d_fft & dconv_pair now guarded.")
# ───────────────────────────────────────────────────────────────────────────
