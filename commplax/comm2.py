# Copyright 2021 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
from scipy import signal, special
import matplotlib.pyplot as plt

# 量子随机比特（可选）
try:
    import quantumrandom
except Exception:  # 如果环境无该库，保留函数接口，但在调用时报错
    quantumrandom = None

# 依赖 commplax.op 的分帧函数（若未安装 commplax，请根据需要替换为自定义实现）
try:
    from commplax import op
except Exception:
    op = None

# ===========================
# QAM 基础与映射/判决/解调
# ===========================

quasigray_32xqam = np.array([
    -3.+5.j, -1.+5.j, -3.-5.j, -1.-5.j, -5.+3.j, -5.+1.j, -5.-3.j, -5.-1.j,
    -1.+3.j, -1.+1.j, -1.-3.j, -1.-1.j, -3.+3.j, -3.+1.j, -3.-3.j, -3.-1.j,
     3.+5.j,  1.+5.j,  3.-5.j,  1.-5.j,  5.+3.j,  5.+1.j,  5.-3.j,  5.-1.j,
     1.+3.j,  1.+1.j,  1.-3.j,  1.-1.j,  3.+3.j,  3.+1.j,  3.-3.j,  3.-1.j
], dtype=np.complex128)

def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1) == 0)

def is_square_qam(L: int) -> bool:
    return is_power_of_two(L) and (int(np.log2(L)) % 2 == 0)

def is_cross_qam(L: int) -> bool:
    return is_power_of_two(L) and (int(np.log2(L)) % 2 == 1)

def grayenc_int(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=int)
    return x ^ (x >> 1)

def graydec_int(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=int))
    mask = np.array(x)
    while mask.any():
        I = mask > 0
        mask[I] >>= 1
        x[I] ^= mask[I]
    return x

def square_qam_grayenc_int(x: np.ndarray, L: int) -> np.ndarray:
    """
    参考：Wesel et al., IEEE TIT 2001, Constellation labeling for linear encoders.
    """
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    B = int(np.log2(M))
    x1 = x // M
    x2 = x %  M
    return (grayenc_int(x1) << B) + grayenc_int(x2)

def square_qam_graydec_int(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    B = int(np.log2(M))
    x1 = graydec_int(x >> B)
    x2 = graydec_int(x % (1 << B))
    return x1 * M + x2

def parseqamorder(type_str: str) -> int:
    if type_str.lower() == 'qpsk':
        type_str = '4QAM'
    M = int(re.findall(r'\d+', type_str)[0])
    T = re.findall(r'[a-zA-Z]+', type_str)[0].lower()
    if T != 'qam':
        raise ValueError(f'{T} is not implemented yet')
    return M

def const(type_str=None, norm: bool = False) -> np.ndarray:
    """按自然命名或阶数生成星座点；支持方形 QAM 与 32-cross-QAM。"""
    if isinstance(type_str, str):
        L = parseqamorder(type_str)
    else:
        L = int(type_str)
    if is_square_qam(L):
        A = np.linspace(-np.sqrt(L) + 1, np.sqrt(L) - 1, int(np.sqrt(L)), dtype=np.float64)
        C = (A[None, :] + 1j * A[::-1, None]).reshape(-1)
    elif L == 32:
        C = quasigray_32xqam.copy()
    else:
        raise ValueError(f'Only square-QAM and 32-cross-QAM are supported, got {L}')
    if norm:
        C /= np.sqrt(2 * (L - 1) / 3)
    return C.astype(np.complex128)

def pamdecision(x: np.ndarray, M: int) -> np.ndarray:
    """一维 PAM 判决（网格步长 2，边界截断）。"""
    x = np.asarray(x)
    y = np.atleast_1d((np.round(x / 2 + 0.5) - 0.5) * 2).astype(int)
    bd = M - 1
    y[y >  bd] =  bd
    y[y < -bd] = -bd
    return y

def square_qam_decision(x: np.ndarray, L: int) -> np.ndarray:
    x = np.atleast_1d(x)
    M = int(np.sqrt(L))
    if np.iscomplexobj(x):
        I = pamdecision(np.real(x), M)
        Q = pamdecision(np.imag(x), M)
        y = I + 1j * Q
    else:
        I = pamdecision(x[0], M)
        Q = pamdecision(x[1], M)
        y = (I, Q)
    return y.astype(np.complex128)

def cross_qam_decision(x: np.ndarray, L: int, return_int: bool = False):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    c = const(L)
    idx = np.argmin(np.abs(x[:, None] - c[None, :])**2, axis=1)
    y = c[idx]
    return idx if return_int else y

def qamdecision(x: np.ndarray, L: int) -> np.ndarray:
    return square_qam_decision(x, L) if is_square_qam(L) else cross_qam_decision(x, L)

def square_qam_mod(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    A = np.linspace(-M + 1, M - 1, M, dtype=np.float64)
    C = A[None,:] + 1j * A[::-1, None]
    d = square_qam_graydec_int(x, L)
    return (C[d // M, d % M]).astype(np.complex128)

def cross_qam_mod(x: np.ndarray, L: int) -> np.ndarray:
    if L == 32:
        return quasigray_32xqam[np.asarray(x, dtype=int)]
    raise ValueError(f'Cross QAM size {L} not implemented')

def qammod(x: np.ndarray, L: int) -> np.ndarray:
    return square_qam_mod(x, L) if is_square_qam(L) else cross_qam_mod(x, L)

def square_qam_demod(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x)
    M = int(np.sqrt(L))
    x = square_qam_decision(x, L)
    c = ((np.real(x) + M - 1) // 2).astype(int)
    r = ((M - 1 - np.imag(x)) // 2).astype(int)
    return square_qam_grayenc_int(r * M + c, L)

def qamdemod(x: np.ndarray, L: int) -> np.ndarray:
    return square_qam_demod(x, L) if is_square_qam(L) else cross_qam_decision(x, L, return_int=True)

def int2bit(d: np.ndarray, W: int) -> np.ndarray:
    """
    把整数标签转 bit。W 通常取 √L（历史口径），后续可用 active mask 选取真正的 m=log2(L) 位。
    """
    W = int(W)
    d = np.atleast_1d(d).astype(np.uint8)
    b = np.unpackbits(d[:, None], axis=1)[:, -W:]
    return b.astype(np.uint8)

def bit2int(b: np.ndarray, M: int) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8)
    d = np.packbits(np.pad(b.reshape((-1, M)), ((0, 0), (8 - M, 0))))
    return d

def grayqamplot(L: int):
    """画出 Gray 标记的 QAM 星座（小规模 L 时可视化）。"""
    M = int(np.log2(L))
    x = list(range(L))
    y = qammod(x, L)
    fstr = "{:0" + str(M) + "b}"
    I = np.real(y); Q = np.imag(y)
    plt.figure(num=None, figsize=(8, 6), dpi=100)
    plt.axis('equal')
    plt.scatter(I, Q, s=16)
    for i in range(L):
        plt.annotate(fstr.format(x[i]), (I[i], Q[i]))

def randpam(s: int, n: int, p=None):
    m = int(s)
    a = np.linspace(-m + 1, m - 1, m, dtype=np.float64)
    return np.random.choice(a, n, p=p) + 1j * np.random.choice(a, n, p=p)

def randqam(s: int, n: int, p=None):
    m = int(np.sqrt(s))
    a = np.linspace(-m + 1, m - 1, m, dtype=np.float64)
    return np.random.choice(a, n, p=p) + 1j * np.random.choice(a, n, p=p)

def anuqrng_bit(L: int):
    """https://github.com/lmacken/quantumrandom"""
    if quantumrandom is None:
        raise RuntimeError("quantumrandom 未安装，无法获取量子随机比特")
    L    = int(L)
    N    = 0
    bits = []
    while N < L:
        b = np.unpackbits(np.frombuffer(quantumrandom.binary(), dtype=np.uint8))
        N += len(b)
        bits.append(b)
    bits = np.concatenate(bits)[:L]
    return bits

# ===========================
# 通用信号处理工具
# ===========================

def shape_signal(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x))
    if x.ndim == 1:
        x = x[..., None]
    return x

def getpower(x: np.ndarray, real: bool = False) -> float:
    return np.mean(x.real**2, axis=0) + 1j * np.mean(x.imag**2, axis=0) if real else np.mean(np.abs(x)**2, axis=0)

def normpower(x: np.ndarray, real: bool = False):
    if real:
        p = getpower(x, real=True)
        return x.real / np.sqrt(p.real) + 1j * x.imag / np.sqrt(p.imag)
    else:
        return x / np.sqrt(getpower(x))

def delta(taps, dims=None, dtype=np.complex64):
    mf = np.zeros(taps, dtype=dtype)
    mf[(taps - 1) // 2] = 1.
    return mf if dims is None else np.tile(mf[:, None], dims)

def gauss_kernel(n=11, sigma=1, dims=None, dtype=np.complex64):
    r = np.arange(-int(n / 2), int(n / 2) + 1) if n % 2 else np.linspace(-int(n / 2) + 0.5, int(n / 2) - 0.5, n)
    w = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x)**2 / (2 * sigma**2)) for x in r]).astype(dtype)
    return w if dims is None else np.tile(w[:, None], dims)

def gauss_minbw(taps):
    return 1 / (2 * np.pi * ((taps + 1) / 6) * 1.17741)

def gauss(bw, taps=None, oddtaps=True, dtype=np.float64):
    """https://en.wikipedia.org/wiki/Gaussian_filter"""
    eps = 1e-8
    gamma = 1 / (2 * np.pi * bw * 1.17741)
    mintaps = int(np.ceil(6 * gamma - 1 - eps))
    if taps is None:
        taps = mintaps
    elif taps < mintaps:
        raise ValueError(f'required {taps} taps which is less than minimal default {mintaps}')
    if oddtaps is not None:
        if oddtaps:
            taps = mintaps if mintaps % 2 == 1 else mintaps + 1
        else:
            taps = mintaps if mintaps % 2 == 0 else mintaps + 1
    return gauss_kernel(taps, gamma, dtype=dtype)

def rcosdesign(beta, span, sps, shape='normal', dtype=np.float64):
    """RRC/RC 滤波器（对齐 MATLAB rcosdesign 口径）。"""
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1, dtype=dtype) / sps
    b = np.zeros_like(t)
    eps = np.finfo(dtype).eps
    if beta == 0:
        beta = np.finfo(dtype).tiny
    if shape == 'normal':
        denom = 1 - (2 * beta * t) ** 2
        ind1 = np.where(abs(denom) > np.sqrt(eps), True, False)
        ind2 = ~ind1
        b[ind1] = np.sinc(t[ind1]) * (np.cos(np.pi * beta * t[ind1]) / denom[ind1]) / sps
        b[ind2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)
    elif shape == 'sqrt':
        ind1 = np.where(t == 0, True, False)
        ind2 = np.where(abs(abs(4 * beta * t) - 1.0) < np.sqrt(eps), True, False)
        ind3 = ~(ind1 | ind2)
        b[ind1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        b[ind2] = (
            1 / (2 * np.pi * sps)
            * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
            - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
            + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))
        )
        b[ind3] = (
            -4 * beta / sps * (np.cos((1 + beta) * np.pi * t[ind3]) +
                               np.sin((1 - beta) * np.pi * t[ind3]) / (4 * beta * t[ind3]))
            / (np.pi * ((4 * beta * t[ind3])**2 - 1))
        )
    else:
        raise ValueError('invalid shape')
    b /= np.sqrt(np.sum(b**2))  # normalize filter gain
    return b

def upsample(x, n, axis=0, trim=False):
    x = np.atleast_1d(x)
    x = signal.upfirdn([1], x, n, axis=axis)
    pads = np.zeros((x.ndim, 2), dtype=int)
    pads[axis, 1] = n - 1
    y = x if trim else np.pad(x, pads)
    return y

def resample(x, p, q, axis=0):
    p = int(p); q = int(q)
    gcd = np.gcd(p, q)
    return signal.resample_poly(x, p//gcd, q//gcd, axis=axis)

def qamscale(modformat):
    if isinstance(modformat, str):
        M = parseqamorder(modformat)
    else:
        M = modformat
    return np.sqrt((M - 1) * 2 / 3) if is_square_qam(M) else np.sqrt(2/3 * (M * 31/32 - 1))

def dbp_params(
    sample_rate, span_length, spans, freqs,
    launch_power=0, steps_per_span=1, virtual_spans=None,
    carrier_frequency=194.1e12, fiber_dispersion=16.7E-6, fiber_dispersion_slope=0.08e3,
    fiber_loss=.2E-3, fiber_core_area=80E-12, fiber_nonlinear_index=2.6E-20,
    fiber_reference_frequency=194.1e12, ignore_beta3=False, polmux=True,
    domain='time', step_method="uniform"):
    """生成 DBP（数字反向传输）参数（与原始实现等价）。"""
    domain = domain.lower()
    assert domain in ('time', 'frequency')
    pi  = np.pi
    log = np.log
    exp = np.exp
    ifft = np.fft.ifft
    if virtual_spans is None:
        virtual_spans = spans
    C       = 299792458.
    lambda_ = C / fiber_reference_frequency
    B_2     = -fiber_dispersion * lambda_**2 / (2 * pi * C)
    B_3     = 0. if ignore_beta3 else \
        (fiber_dispersion_slope * lambda_**2 + 2 * fiber_dispersion * lambda_) * (lambda_ / (2 * pi * C))**2
    gamma   = 2 * pi * fiber_nonlinear_index / lambda_ / fiber_core_area
    LP      = 10.**(launch_power / 10 - 3)
    alpha   = fiber_loss / (10. / log(10.))
    L_eff   = lambda h: (1 - exp(-alpha * h)) / alpha
    NIter   = virtual_spans * steps_per_span
    delay   = (freqs - 1) // 2
    dw      = 2 * pi * (carrier_frequency - fiber_reference_frequency)
    w_res   = 2 * pi * sample_rate / freqs
    k       = np.arange(freqs)
    w       = np.where(k > delay, k - freqs, k) * w_res  # ifftshifted
    if step_method.lower() == "uniform":
        H   = exp(-1j * (-B_2 / 2 * (w + dw)**2 + B_3 / 6 * (w + dw)**3) *
                  span_length * spans / virtual_spans / steps_per_span)
        H_casual = H * exp(-1j * w * delay / sample_rate)
        h_casual = ifft(H_casual)
        phi = spans / virtual_spans * gamma * L_eff(span_length / steps_per_span) * LP * \
            exp(-alpha * span_length * (steps_per_span - np.arange(0, NIter) % steps_per_span-1) / steps_per_span)
    else:
        raise ValueError(f"step method '{step_method}' not implemented")
    dims = 2 if polmux else 1
    H = np.tile(H[None, :, None], (NIter, 1, dims))
    h_casual = np.tile(h_casual[None, :, None], (NIter, 1, dims))
    phi = np.tile(phi[:, None, None], (1, dims, dims))
    return (h_casual, phi) if domain == 'time' else (H, phi)

def firfreqz(h, sr=1, N=8192, t0=None, bw=None):
    h = np.atleast_2d(h)
    T = h.shape[-1]
    if t0 is None:
        t0 = (T - 1) // 2 + 1
    H = []
    for hi in h:
        w, Hi = signal.freqz(hi, worN=N, whole=True)
        Hi *= np.exp(1j * w * (t0 - 1))
        H.append(Hi)
    H = np.array(H)
    w = (w + np.pi) % (2 * np.pi) - np.pi
    H = np.squeeze(np.fft.fftshift(H, axes=-1))
    w = np.fft.fftshift(w, axes=-1) * sr / 2 / np.pi
    if bw is not None:
        s = int((sr - bw) / sr / 2 * len(w))
        w = w[s: -s]
        H = H[..., s: -s]
    return w, H

# ===========================
# 对齐/时延
# ===========================

def finddelay(x, y):
    """估计 y 相对 x 的延时（与 MATLAB finddelay 类似，含极端处理）。"""
    x = np.asarray(x); y = np.asarray(y)
    c = np.abs(signal.correlate(x, y, mode='full', method='fft'))
    k = np.arange(-len(y) + 1, len(x))
    i = np.lexsort((np.abs(k), -c))[0]
    d = -k[i]
    return d

def align_periodic(y, x, begin=0, last=2000, thr=0.5):
    """简易周期对齐：返回对齐后的 x 以及每极化/通道的移位量。"""
    dims = x.shape[-1]
    z = np.zeros_like(x)
    def step(v, u):
        c = np.abs(signal.correlate(u, v[begin:begin+last], mode='full', method='fft'))
        c /= np.max(c)
        k = np.arange(-len(x) + 1, len(y))
        idx = np.where(c > thr)[0]
        idx = idx[np.argsort(np.atleast_1d(c[idx]))[::-1]]
        j = -k[idx] + begin + last
        return j
    r0 = step(y[:, 0], x[:, 0])
    if dims > 1:
        if len(r0) == 1:
            r0 = r0[0]
            r1 = step(y[:, 1], x[:, 1])[0]
        elif len(r0) == 2:
            r1 = r0[1]; r0 = r0[0]
        else:
            raise RuntimeError('bad input')
        z[:, 0] = np.roll(x[:, 0], r0)
        z[:, 1] = np.roll(x[:, 1], r1)
        d = np.stack((r0, r1))
    else:
        z[:, 0] = np.roll(x[:, 0], r0)
        d = r0
    z = np.tile(z, (len(y)//len(z)+1, 1))[:len(y), :]
    return z, d

# ===========================
# HD（硬判）指标
# ===========================

def qamqot(y: np.ndarray, x: np.ndarray, count_dim=True, count_total=True,
           L=None, eval_range=(0, 0), scale=1.0) -> pd.DataFrame:
    """
    经典 HD 指标：BER / Q_dB / SNR（用判决口径一致的 demod）
    """
    assert y.shape[0] == x.shape[0]
    y = y[eval_range[0]: y.shape[0] + (eval_range[1] if eval_range[1] <= 0 else eval_range[1])] * scale
    x = x[eval_range[0]: x.shape[0] + (eval_range[1] if eval_range[1] <= 0 else eval_range[1])] * scale
    y = shape_signal(y); x = shape_signal(x)
    p = np.rint(x.real) + 1j * np.rint(x.imag)
    if np.max(np.abs(p - x)) > 1e-2:
        raise ValueError('the scaled x is seemly not canonical')
    if L is None:
        L = len(np.unique(p))
    D = y.shape[-1]
    z = [(a, b) for a, b in zip(y.T, x.T)]
    snr_fn = lambda yy, xx: 10. * np.log10(getpower(xx, False) / (getpower(xx - yy, False) + 1e-18))
    def _f(pair):
        yy, xx = pair
        M = int(np.sqrt(L))
        by = int2bit(qamdemod(yy, L), M).ravel()
        bx = int2bit(qamdemod(xx, L), M).ravel()
        BER = np.count_nonzero(by - bx) / len(by)
        with np.errstate(divide='ignore'):
            QSq = 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * BER), 0.))
        SNR = snr_fn(yy, xx)
        return BER, QSq, SNR
    qot, ind = [], []
    if count_dim:
        qot += list(map(_f, z)); ind += [f'dim{n}' for n in range(D)]
    if count_total:
        qot += [_f((y.ravel(), x.ravel()))]; ind += ['total']
    return pd.DataFrame(qot, columns=['BER', 'QSq', 'SNR'], index=ind)

def qamqot_local(y, x, frame_size=10000, L=None, scale=1, eval_range=None):
    """局部（分帧）QOT 指标，返回每一帧（上采样展平）的 BER/QSq/SNR。"""
    if op is None:
        raise RuntimeError("需要 commplax.op.frame 支持；请安装 commplax 或替换为你的分帧实现")
    y = shape_signal(y)
    x = shape_signal(x)
    if L is None:
        L = len(np.unique(x))
    Y = op.frame(y, frame_size, frame_size, True)
    X = op.frame(x, frame_size, frame_size, True)
    zf = [(yf, xf) for yf, xf in zip(Y, X)]
    f = lambda z: qamqot(z[0], z[1], count_dim=True, L=L, scale=scale).to_numpy()
    qot_local = np.stack(list(map(f, zf)))
    qot_local_ip = np.repeat(qot_local, frame_size, axis=0) # 简单重复插值
    return {'BER': qot_local_ip[...,0], 'QSq': qot_local_ip[...,1], 'SNR': qot_local_ip[...,2]}

def corr_local(y, x, frame_size=10000, L=None):
    """分帧相关性（幅度）。"""
    if op is None:
        raise RuntimeError("需要 commplax.op.frame 支持；请安装 commplax 或替换为你的分帧实现")
    y = shape_signal(y)
    x = shape_signal(x)
    if L is None:
        L = len(np.unique(x))
    Y = op.frame(y, frame_size, frame_size, True)
    X = op.frame(x, frame_size, frame_size, True)
    zf = [(yf, xf) for yf, xf in zip(Y, X)]
    f = lambda z: np.abs(np.sum(z[0] * z[1].conj(), axis=0))
    qot_local = np.stack(list(map(f, zf)))
    qot_local_ip = np.repeat(qot_local, frame_size, axis=0)
    return qot_local_ip

def snrstat(y, x, frame_size=10000, L=None, eval_range=(0, 0), scale=1):
    """局部 SNR 的均值/方差/上下偏差等统计。"""
    assert y.shape[0] == x.shape[0]
    y = y[eval_range[0]: y.shape[0] + (eval_range[1] if eval_range[1] <= 0 else eval_range[1])] * scale
    x = x[eval_range[0]: x.shape[0] + (eval_range[1] if eval_range[1] <= 0 else eval_range[1])] * scale
    snr_local = qamqot_local(y, x, frame_size, L)['SNR'][:, :2]
    sl_mean = np.mean(snr_local, axis=0)
    sl_std = np.std(snr_local, axis=0)
    sl_max = np.max(snr_local, axis=0)
    sl_min = np.min(snr_local, axis=0)
    return np.stack((sl_mean, sl_std, sl_mean - sl_min, sl_max - sl_mean))

# ===========================
# SD（软判）流程：估噪→LLR→校准→温度→量化→（可选）解码→GMI
# ===========================

def _whiten_cov(n_reim: np.ndarray):
    """n_reim: shape [2, N]（第一行实部、第二行虚部）"""
    Sigma = np.cov(n_reim)
    sigma2 = 0.5 * float(np.trace(Sigma))
    return Sigma, sigma2

def _alpha_ls(y: np.ndarray, s: np.ndarray, eps: float = 1e-12) -> complex:
    num = np.vdot(s, y)         # s^H y
    den = np.vdot(s, s) + eps   # s^H s
    return num / den

def estimate_noise_stats(y: np.ndarray, L: int, use_truth: np.ndarray = None):
    """
    决策导向/真值导向估噪：估 α、残差协方差 Σ、标量方差 σ²、以及符号估计 hat{s}
    """
    if use_truth is not None:
        s_hat = np.asarray(use_truth, dtype=np.complex128).reshape(-1)
    else:
        s_hat = qamdecision(y, L).astype(np.complex128).reshape(-1)
    alpha = _alpha_ls(y, s_hat)
    r = y - alpha * s_hat
    Sigma, sigma2 = _whiten_cov(np.vstack([r.real, r.imag]))
    return alpha, Sigma, sigma2, s_hat

def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)), axis=axis)

def _build_const_and_labels(L: int):
    """
    在星座点上反推与解调口径一致的 bit 标签。
    历史位宽 W=√L，随后用 active mask 取真正的 m=log2(L) 位。
    """
    const_pts = const(L)
    W = int(np.sqrt(L))
    dec_ints = qamdemod(const_pts, L)
    labels_full = int2bit(dec_ints, W)  # [L, W]
    col_sum = labels_full.sum(axis=0)
    active = (col_sum > 0) & (col_sum < len(labels_full))
    labels = labels_full[:, active]     # [L, m]
    return const_pts.astype(np.complex128), labels.astype(np.uint8), active

def _llr_awgn(y: np.ndarray, const_pts: np.ndarray, bitlabels: np.ndarray,
              sigma2: float, Sigma: np.ndarray = None) -> np.ndarray:
    """
    返回按位 LLR（N×m），定义：LLR = log p(y|b=1) - log p(y|b=0)
    Sigma=None → 欧氏度量；否则用马氏距离（椭圆度量）。
    """
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    N = y.shape[0]
    m = bitlabels.shape[1]
    if Sigma is not None:
        Sinv = np.linalg.inv(Sigma)
        y_xy = np.stack([y.real, y.imag], axis=1)[:, None, :]                  # [N,1,2]
        c_xy = np.stack([const_pts.real, const_pts.imag], axis=1)[None, :, :]  # [1,L,2]
        diff = y_xy - c_xy
        d2 = np.einsum('...i,ij,...j->...', diff, Sinv, diff)                  # [N,L]
        ll = -0.5 * d2
    else:
        d2 = np.abs(y.reshape(-1, 1) - const_pts.reshape(1, -1))**2            # [N,L]
        ll = -d2 / (sigma2 + 1e-18)
    llrs = np.zeros((N, m), dtype=np.float64)
    for i in range(m):
        mask1 = (bitlabels[:, i] == 1)
        mask0 = ~mask1
        lse1 = _logsumexp(ll[:, mask1], axis=1)
        lse0 = _logsumexp(ll[:, mask0], axis=1)
        llrs[:, i] = lse1 - lse0
    return llrs

def calibrate_llr_polarity(llr: np.ndarray, bits: np.ndarray):
    """逐位极性自校正（评估可用；上线建议用导频锁定极性）。"""
    agree = ((llr > 0).astype(np.uint8) == bits.astype(np.uint8)).mean(axis=0)  # [m]
    flip = (agree < 0.5)
    llr[:, flip] = -llr[:, flip]
    return llr, flip

def find_best_temperature(llr: np.ndarray, bits: np.ndarray,
                          grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    best_t, best_g = 1.0, -1e9
    for t in grid:
        g = _gmi_from_llr_bits(llr * t, bits)
        if g > best_g:
            best_g, best_t = g, t
    return best_t, best_g

def quantize_llr(llr: np.ndarray, bitwidth: int = 6, method: str = 'std', k: float = 3.0):
    """
    定点量化到整数（默认 int8），返回 (llr_q, scale)。
    method='std'：以列标准差设定动态范围，使 ±kσ ≈ 最大码值。
    """
    maxq = (1 << (bitwidth - 1)) - 1
    if method == 'std':
        s = np.std(llr, axis=0) + 1e-12
        scale = maxq / (k * s)                  # broadcast 到列
        llr_q = np.clip(np.round(llr * scale), -maxq, maxq).astype(np.int8)
        return llr_q, scale
    raise ValueError("unknown quant method")

def _gmi_from_llr_bits(llrs: np.ndarray, bits: np.ndarray) -> float:
    """
    per-bit MI = 1 - E[ log2(1 + exp(-(2b-1)*L)) ]；GMI_total = m * per-bit MI
    """
    m = llrs.shape[1]
    s = 2 * bits.astype(np.int8) - 1
    x = - s * llrs
    t = np.maximum(0.0, x) + np.log1p(np.exp(-np.abs(x)))  # softplus
    per_bit_mi = 1.0 - float(np.mean(t)) / np.log(2.0)
    return float(m * per_bit_mi)

def evaluate_sd(y: np.ndarray, x_ref: np.ndarray, L: int,
                decoder=None,
                use_oracle_noise: bool = True,
                elliptical_llr: bool = True,
                temp_grid=(0.75, 1.0, 1.25),
                bitwidth: int = 6,
                return_artifacts: bool = False):
    """
    软判（SD）评估主函数：同一份 LLR → GMI/NGMI；可选调用软解码器得到 post-FEC。
    - decoder: callable(llr_q) -> {'post_fec_ber': ...} 或 None
    """
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.complex128).reshape(-1)
    assert y.shape == x_ref.shape

    # 1) 估噪（oracle/决策导向）
    alpha, Sigma, sigma2, s_hat = estimate_noise_stats(y, L, use_truth=x_ref if use_oracle_noise else None)

    # 2) 构建星座与位标签（与解调口径一致）
    const_pts, bitlabels, active_mask = _build_const_and_labels(L)
    m = bitlabels.shape[1]

    # 3) LLR（椭圆或欧氏）
    if elliptical_llr:
        llr = _llr_awgn(y, const_pts, bitlabels, sigma2, Sigma)
    else:
        llr = _llr_awgn(y, const_pts, bitlabels, sigma2, None)

    # 4) 真比特（评估使用；上线用导频）
    W = int(np.sqrt(L))
    bits_full = int2bit(qamdemod(x_ref, L), W)
    bits = bits_full[:, active_mask].astype(np.uint8)

    # 5) 逐位极性校准（评估方便；上线请用导频做极性锁）
    llr, flip_mask = calibrate_llr_polarity(llr, bits)

    # 6) 温度扫描（也可改为学习/估计）
    t_opt, _ = find_best_temperature(llr, bits, grid=temp_grid)
    llr_cal = llr * t_opt

    # 7) 定点量化（与解码器位宽一致）
    llr_q, scale = quantize_llr(llr_cal, bitwidth=bitwidth, method='std', k=3.0)

    # 8) （可选）软解码
    post = {}
    if decoder is not None:
        post = decoder(llr_q)  # 建议在 adapter 内处理交织/帧化/穿孔

    # 9) 用“同一份（温度校准后的）LLR”计算 GMI/NGMI
    gmi = _gmi_from_llr_bits(llr_cal.astype(np.float64), bits)
    ngmi = gmi / m

    # 10) 同时给出 HD 指标以对照
    hd_df = qamqot(y, x_ref, L=L, count_dim=False, count_total=True)
    pre_ber = float(hd_df.loc['total', 'BER'])
    pre_qdb = float(hd_df.loc['total', 'QSq'])

    out = {
        'GMI_bits_per_dim': gmi,
        'NGMI': ngmi,
        'pre_BER': pre_ber,
        'pre_Q_dB': pre_qdb,
        'temp': t_opt,
        'alpha': alpha
    }
    out.update(post)

    if return_artifacts:
        out.update(dict(
            llr=llr, llr_cal=llr_cal, llr_q=llr_q,
            Sigma=Sigma, sigma2=sigma2, flip_mask=flip_mask, scale=scale
        ))
    return out

def evaluate_hd_and_sd(y: np.ndarray, x_ref: np.ndarray, L: int,
                       decoder=None,
                       sd_kwargs: dict = None):
    """便捷入口：返回 {'HD': DataFrame(qamqot), 'SD': dict}。"""
    hd = qamqot(y, x_ref, L=L, count_dim=False, count_total=True)
    sd = evaluate_sd(y, x_ref, L, decoder=decoder, **(sd_kwargs or {}))
    return {'HD': hd, 'SD': sd}
