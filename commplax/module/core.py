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
import jax
import numpy as np
from jax import numpy as jnp
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import Scope, lift, freeze, unfreeze
from commplax import comm, xcomm, xop, adaptive_filter as af
from commplax.util import wrapped_partial as wpartial
from typing import Any, NamedTuple, Iterable, Callable, Optional
from jax import random
from jax import lax
from flax import linen as nn
from flax.core import Scope, init, apply
from typing import Tuple
from functools import partial
Array = Any
from jax import debug
from jax.nn import sigmoid
# related: https://github.com/google/jax/issues/6853
@struct.dataclass
class SigTime:
  start: int = struct.field(pytree_node=False)
  stop: int = struct.field(pytree_node=False)
  sps: int = struct.field(pytree_node=False)

class Signal(NamedTuple):
    val: Array
    t: Any = SigTime(0, 0, 2)

    def taxis(self):
        return self.t[0].shape[0], -self.t[0].shape[1]

    def __mul__(self, other):
        Signal._check_type(other)
        return Signal(self.val * other, self.t)

    def __add__(self, other):
        Signal._check_type(other)
        return Signal(self.val + other, self.t)
        

    def __sub__(self, other):
        Signal._check_type(other)
        return Signal(self.val - other, self.t)

    def __truediv__(self, other):
        Signal._check_type(other)
        return Signal(self.val / other, self.t)

    def __floordiv__(self, other):
        Signal._check_type(other)
        return Signal(self.val // other, self.t)

    def __imul__(self, other):
        return self * other

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    @classmethod
    def _check_type(cls, other):
        assert not isinstance(other, cls), 'not implemented'
      
def zeros(key, shape, dtype=jnp.float32): return jnp.zeros(shape, dtype)
def ones(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)
def delta(key, shape, dtype=jnp.float32):
    k1d = comm.delta(shape[0], dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])
def gauss(key, shape, dtype=jnp.float32):
    taps = shape[0]
    k1d = comm.gauss(comm.gauss_minbw(taps), taps=taps, dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])


def dict_replace(col, target, leaf_only=True):
    col_flat = flatten_dict(unfreeze(col))
    diff = {}
    for keys_flat in col_flat.keys():
        for tar_key, tar_val in target.items():
            if (keys_flat[-1] == tar_key if leaf_only else (tar_key in keys_flat)):
                diff[keys_flat] = tar_val
    col_flat.update(diff)
    col = unflatten_dict(col_flat)
    return col


def update_varcolbykey(var, col_name, target, leaf_only=True):
    wocol, col = var.pop(col_name)
    col = dict_replace(col, target, leaf_only=leaf_only)
    del var
    return freeze({**wocol, col_name: col})


def update_aux(var, tar):
    return update_varcolbykey(var, 'aux_inputs', tar, leaf_only=True)


def conv1d_t(t, taps, rtap, stride, mode):
    assert t.sps >= stride, f'sps of input SigTime must be >= stride: {stride}, got {t.sps} instead'
    if rtap is None:
        rtap = (taps - 1) // 2
    delay = -(-(rtap + 1) // stride) - 1
    if mode == 'full':
        tslice = (-delay * stride, taps - stride * (rtap + 1)) #TODO: think more about this
    elif mode == 'same':
        tslice = (0, 0)
    elif mode == 'valid':
        tslice = (delay * stride, (delay + 1) * stride - taps)
    else:
        raise ValueError('invalid mode {}'.format(mode))
    return SigTime((t.start + tslice[0]) // stride, (t.stop + tslice[1]) // stride, t.sps // stride)


def conv1d_slicer(taps, rtap=None, stride=1, mode='valid'):
    def slicer(signal):
        x, xt = signal
        yt = conv1d_t(xt, taps, rtap, stride, mode)
        D = xt.sps // yt.sps
        zt = SigTime(yt.start * D, yt.stop * D, xt.sps)
        x = x[zt.start - xt.start: x.shape[0] + zt.stop - xt.stop]
        return Signal(x, zt)
    return slicer


def fullsigval(inputs: Signal, fill_value=1):
    x, t = inputs
    full_shape = (x.shape[0] + t.start - t.stop,) + x.shape[1:]
    return jnp.full(full_shape, fill_value, dtype=x.dtype)


def vmap(f,
         variable_axes={
             'params': -1,
             'const': None
         },
         split_rngs={
             'params': True,
         },
         in_axes=(Signal(-1, None),), out_axes=Signal(-1, None)):
    # in_axes needs to be wrapped by a tuple, see Flax's lifted vmap implemetation:
    # https://github.com/google/flax/blob/82e9798274c927286878c4600b4b09650d1e7935/flax/core/lift.py#L395
    vf = lift.vmap(f,
                   variable_axes=variable_axes, split_rngs=split_rngs,
                   in_axes=in_axes, out_axes=out_axes)
    vf.__name__ = 'vmapped_' + f.__name__ # [Workaround]: lifted transformation does not keep the original name
    return vf


def scan(f, in_axes=0, out_axes=0):
    sf = lift.scan(f, in_axes=in_axes, out_axes=out_axes)
    sf.__name__ = 'scanned' + f.__name__
    return sf


def simplefn(scope, signal, fn=None, aux_inputs=None):
    assert fn is not None, 'simple function cannot be None'
    aux = ()
    if aux_inputs is not None:
        aux_name, aux_init = aux_inputs
        aux += scope.variable('aux_inputs', aux_name, aux_init, signal).value,
    return fn(signal, *aux)


def batchpowernorm(scope, signal, momentum=0.999, mode='train'):
    running_mean = scope.variable('norm', 'running_mean',
                                  lambda *_: 0. + jnp.ones(signal.val.shape[-1]), ())
    if mode == 'train':
        mean = jnp.mean(jnp.abs(signal.val)**2, axis=0)
        running_mean.value = momentum * running_mean.value + (1 - momentum) * mean
    else:
        mean = running_mean.value
    return signal / jnp.sqrt(mean)


# ====== Debug switches ======
DEBUG_VALIDATE = True     # 开启验证打印
DEBUG_MAXPRINT = 3        # 每个循环最多打印几次（避免刷屏）

# ====== 小工具：相位规约到 [-pi, pi] ======
def _wrap_to_pi(phi: Array) -> Array:
    return (phi + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def _chk_array(label: str, x: Array):
    return _check_array(label, x)

# ---- 复数安全护栏：先清 NaN/Inf，再分别裁剪实部/虚部 ----
def _guard_complex(x, clip: float = 1e3):
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    xr = jnp.clip(jnp.real(x), -clip, clip)
    xi = jnp.clip(jnp.imag(x), -clip, clip)
    return xr + 1j * xi


# ====== 小工具：核体检（检查 |H|≈1、tap幅度） ======
def _check_kernel(label: str, h: Array):
    if not DEBUG_VALIDATE:
        return
    H = jnp.fft.fft(h)
    amp_err_max = jnp.max(jnp.abs(jnp.abs(H) - 1.0))
    tap_max = jnp.max(jnp.abs(h))
    tap_l1  = jnp.sum(jnp.abs(h))
    debug.print("[CHK] {label}: max||H|-1|={:.3e}, max|h|={:.3e}, sum|h|={:.3e}",
                amp_err_max, tap_max, tap_l1, label=label)

# ====== 小工具：打印一次工况（首轮即可） ======
def _log_aux_once(scope: Scope):
    if not DEBUG_VALIDATE:
        return
    # 只在 init 阶段成功创建；apply 阶段如果没创建过则静默返回
    if scope.get_variable('const', '_aux_logged') is None and scope.is_mutable_collection('const'):
        fs    = _aux(scope, 'fs', 1.0)
        dz    = _aux(scope, 'dz', 1.0)
        b2    = _aux(scope, 'beta2', 0.0)
        b3    = _aux(scope, 'beta3', 0.0)
        dw    = _aux(scope, 'dw', 0.0)
        sgn   = _aux(scope, 'lin_sign', 1.0)
        ign3  = _aux(scope, 'ignore_beta3', 0.0)
        ptx   = _aux(scope, 'launch_power', 1.0)
        p_nei = _aux(scope, 'sum_neigh_power', 0.0)
        dfmin = _aux(scope, 'min_ch_spacing', 0.0)
        z     = _cond_z(scope)
        debug.print(
            "[AUX] fs={:.3e} Hz, dz={:.3e} m, beta2={:.3e} s^2/m, beta3={:.3e} s^3/m, dw={:.3e} rad/s, sign={}, ignore_b3={}, "
            "launch_power={:.3e} W, sum_neigh={:.3e} W, min_ch_spacing={:.3e} Hz",
            fs, dz, b2, b3, dw, sgn, ign3, ptx, p_nei, dfmin
        )
        debug.print("[AUX] z vector = {}", z)
        scope.variable('const', '_aux_logged', lambda *_: True, ())


def conv1d(
    scope: Scope,
    signal,
    taps=31,
    rtap=None,
    mode='valid',
    kernel_init=delta,
    conv_fn = xop.convolve):

    x, t = signal
    t = scope.variable('const', 't', conv1d_t, t, taps, rtap, 1, mode).value
    h = scope.param('kernel',
                     kernel_init,
                     (taps,), np.complex64)
    x = conv_fn(x, h, mode=mode)
    return Signal(x, t)
# ===== 工况：在 aux_inputs 里统一注册/读取（有默认值） =====

def _build_phys_defaults():
    """
    这里直接硬编码“默认链路设置”。如果不从 aux_inputs 覆盖，
    D 步就用这套数。数值你可按需要改这张表即可。
    """
    # ---- 你们默认链路（示例：32 Gbaud, sps=2, fs=64e9, 每步=1 span=80 km）----
    fs   = 64e9          # 采样率 [Hz]（例如 2 sps @ 32 Gbaud）
    dz   = 80e3          # 每个 DBP 步长 [m]（例如每步 1 span=80 km）
    fref = 193.4e12      # 参考光频 [Hz]（~1550 nm）
    D_ps_nm_km = 16.7    # 色散参数 D [ps/(nm·km)]
    S_ps_nm2_km = 0.08   # 色散斜率 S [ps/(nm^2·km)]
    # -----------------------------------------------------------------------

    # 把单位换算到 [s^2/m], [s^3/m]
    C   = 299_792_458.0
    pi  = jnp.pi
    lam = C / fref
    # D: ps/(nm·km) -> s/m^2 ；S: ps/(nm^2·km) -> s/m^3
    D   = D_ps_nm_km * 1e-6        # [s/(m^2)] 等价
    S   = S_ps_nm2_km * 1e3        # [s/(m^3)]
    beta2 = -D * lam**2 / (2*pi*C)                         # [s^2/m]
    beta3 = (S * lam**2 + 2*D*lam) * (lam / (2*pi*C))**2   # [s^3/m]

    # 如需以相对 fref 偏移的中心频率 fc，可把下面 dw 改成 2π(fc - fref)
    defaults = {
        'fs'             : float(fs),
        'dz'             : float(dz),
        'beta2'          : float(beta2),
        'beta3'          : float(beta3),
        'dw'             : 0.0,        # 2π(fc - fref) [rad/s]；同载频就填 0
        'lin_sign'       : -1.0,       # -1=DBP 方向；+1=正向传播
        'ignore_beta3'   : 0.0,        # 1.0 可忽略 β3
        'launch_power'   : 1e-3,       # [W] 0 dBm
        'sum_neigh_power': 0.0,
        'min_ch_spacing' : 0.0,
    }
    return defaults


def _get_aux_defaults(scope: Scope):
    """
    取/建 默认工况：
    - 若已存在，可能拿到 Variable 也可能拿到裸值(dict)；两者都要兼容
    - init 阶段可创建 'const/aux_defaults'
    - apply 阶段若不存在，就返回一份字典（不写入变量）
    """
    v = scope.get_variable('const', 'aux_defaults')
    if v is not None:
        return v.value if hasattr(v, 'value') else v  # 兼容两种返回类型
    if scope.is_mutable_collection('const'):
        v = scope.variable('const', 'aux_defaults',
                           lambda *_: _build_phys_defaults(), ())
        return v.value
    return _build_phys_defaults()

def _aux(scope: Scope, key: str, default):
    """
    优先读 aux_inputs[key]；若没有，回落到本文件内置的默认工况表；
    再不行用函数参数的 default。兼容 Variable / 裸值 两种情形。
    """
    v = scope.get_variable('aux_inputs', key)
    if v is not None:
        v = v.value if hasattr(v, 'value') else v
        return jnp.asarray(v)
    defs = _get_aux_defaults(scope)  # dict
    return jnp.asarray(defs.get(key, default))

  
def vmap_share_params(f,
         in_axes=(Signal(-1, None),), out_axes=Signal(-1, None)):
    vf = lift.vmap(f,
                   variable_axes={'params': None, 'const': None},  # 共享参数
                   split_rngs={'params': False},
                   in_axes=in_axes, out_axes=out_axes)
    vf.__name__ = 'vmapped_shared_' + f.__name__
    return vf


def _cond_z(scope: Scope):
    """把可用工况拼成 z；用 float64 先算，最后再安全降到 float32，避免 0*inf→NaN。"""
    fs    = _aux(scope, 'fs', 1.0)
    dz    = _aux(scope, 'dz', 1.0)
    b2    = _aux(scope, 'beta2', 0.0)
    b3    = _aux(scope, 'beta3', 0.0)
    ptx   = _aux(scope, 'launch_power', 1.0)
    p_nei = _aux(scope, 'sum_neigh_power', 0.0)
    dfmin = _aux(scope, 'min_ch_spacing', 0.0)

    # 先转 float64 再做缩放，避免 0*inf→NaN
    b2s = jnp.asarray(b2,   dtype=jnp.float64) * jnp.array(1e27, dtype=jnp.float64)
    b3s = jnp.asarray(b3,   dtype=jnp.float64) * jnp.array(1e39, dtype=jnp.float64)
    dzs = jnp.asarray(dz,   dtype=jnp.float64) * jnp.array(1e-3, dtype=jnp.float64)
    fss = jnp.asarray(fs,   dtype=jnp.float64) * jnp.array(1e-9, dtype=jnp.float64)
    lps = jnp.log10(jnp.maximum(jnp.asarray(ptx, dtype=jnp.float64), 1e-12))
    pns = jnp.asarray(p_nei, dtype=jnp.float64)
    dfs = jnp.asarray(dfmin, dtype=jnp.float64) * jnp.array(1e-9, dtype=jnp.float64)

    z64 = jnp.array([b2s, b3s, dzs, fss, lps, pns, dfs], dtype=jnp.float64)
    # 把可能的 NaN/Inf 拉回到有限数，再降到 float32
    z64 = jnp.nan_to_num(z64, nan=0.0, posinf=1e6, neginf=-1e6)
    return z64.astype(jnp.float32)


def _tiny_mlp(scope: Scope, x: Array, out_dim: int, hidden: int = 16, name: str = 'lin_hyper'):
    W1 = scope.param(name+'/W1', nn.initializers.glorot_uniform(), (x.shape[-1], hidden))
    b1 = scope.param(name+'/b1', nn.initializers.zeros, (hidden,))
    # 关键：W2=0, b2=0 ⇒ 初始 th=0 ⇒ 初始 dphi=0
    W2 = scope.param(name+'/W2', nn.initializers.zeros, (hidden, out_dim))
    b2 = scope.param(name+'/b2', nn.initializers.zeros, (out_dim,))
    return jax.nn.gelu(x @ W1 + b1) @ W2 + b2



def _chebyshev_basis_on_w(w: Array, K: int):
    xi = w / (jnp.max(jnp.abs(w)) + 1e-12)
    B  = []
    T0 = jnp.ones_like(xi); B.append(T0)
    if K == 1: return jnp.stack(B, axis=0)
    T1 = xi; B.append(T1)
    for _ in range(2, K):
        T2 = 2*xi*B[-1] - B[-2]
        B.append(T2)
    return jnp.stack(B, axis=0)  # (K, N)

def _phase_only_kernel(scope: Scope, taps: int, eps: float = 0.05, K: int = 6, dphi_max: float = 1.0):
    fs   = _aux(scope, 'fs', 1.0)
    dz   = _aux(scope, 'dz', 1.0)
    b2   = _aux(scope, 'beta2', 0.0)
    b3   = _aux(scope, 'beta3', 0.0)
    dw   = _aux(scope, 'dw', 0.0)
    sgn  = _aux(scope, 'lin_sign', 1.0)
    ign3 = _aux(scope, 'ignore_beta3', 0.0)

    N      = taps
    delay  = (N - 1) // 2
    w_res  = 2 * jnp.pi * fs / N
    k      = jnp.arange(N)
    w      = jnp.where(k > delay, k - N, k) * w_res

    b3_term  = jnp.where(ign3 > 0.5, 0.0, (b3/6.0) * (w + dw)**3)
    phi_phys = sgn * ( - ( - b2/2.0 * (w + dw)**2 + b3_term ) * dz )  # 物理相位

    B    = _chebyshev_basis_on_w(w, K)              # (K, N)
    z    = _cond_z(scope)                           # (Z,)
    th   = _tiny_mlp(scope, z, out_dim=K, name='lin_hyper')  # (K,)

    # 关键：残差相位做“tanh 限幅”，并给个上限 dphi_max
    raw  = th @ B                                   # (N,)
    dphi = dphi_max * jnp.tanh(raw / jnp.maximum(dphi_max, 1e-6))

    H    = jnp.exp(1j * (phi_phys + eps * dphi))
    Hc   = H * jnp.exp(-1j * w * (delay / fs))
    h    = jnp.fft.ifft(Hc).astype(jnp.complex64)
    return h



def conv1d_phase_cond(scope: Scope, signal, taps=261, mode='valid', eps=0.05, K=6, dphi_max=1.0, conv_fn=xop.convolve):
    x, t = signal
    t2 = scope.variable('const', 't', conv1d_t, t, taps, None, 1, mode).value
    h  = _phase_only_kernel(scope, taps, eps=eps, K=K, dphi_max=dphi_max)
    y  = conv_fn(x, h, mode=mode)
    return Signal(y, t2)





def kernel_initializer(rng, shape):
    return random.normal(rng, shape)  

def mimoconv1d(
    scope: Scope,
    signal,
    taps=31,
    rtap=None,
    dims=2,
    mode='valid',
    kernel_init=zeros,
    conv_fn=xop.convolve):

    x, t = signal
    t = scope.variable('const', 't', conv1d_t, t, taps, rtap, 1, mode).value
    h = scope.param('kernel', kernel_init, (taps, dims, dims), np.float32)
    y = xcomm.mimoconv(x, h, mode=mode, conv=conv_fn)
    return Signal(y, t)
      
      
def mimofoeaf(scope: Scope,
              signal,
              framesize=100,
              w0=0,
              train=False,
              preslicer=lambda x: x,
              foekwargs={},
              mimofn=af.rde,
              mimokwargs={},
              mimoinitargs={}):

    sps = 2
    dims = 2
    tx = signal.t
    # MIMO
    slisig = preslicer(signal)
    auxsig = scope.child(mimoaf,
                         mimofn=mimofn,
                         train=train,
                         mimokwargs=mimokwargs,
                         mimoinitargs=mimoinitargs,
                         name='MIMO4FOE')(slisig)
    y, ty = auxsig # assume y is continuous in time
    yf = xop.frame(y, framesize, framesize)

    foe_init, foe_update, _ = af.array(af.frame_cpr_kf, dims)(**foekwargs)
    state = scope.variable('af_state', 'framefoeaf',
                           lambda *_: (0., 0, foe_init(w0)), ())
    phi, af_step, af_stats = state.value

    af_step, (af_stats, (wf, _)) = af.iterate(foe_update, af_step, af_stats, yf)
    wp = wf.reshape((-1, dims)).mean(axis=-1)
    w = jnp.interp(jnp.arange(y.shape[0] * sps) / sps,
                   jnp.arange(wp.shape[0]) * framesize + (framesize - 1) / 2, wp) / sps
    psi = phi + jnp.cumsum(w)
    state.value = (psi[-1], af_step, af_stats)

    # apply FOE to original input signal via linear extrapolation
    psi_ext = jnp.concatenate([w[0] * jnp.arange(tx.start - ty.start * sps, 0) + phi,
                               psi,
                               w[-1] * jnp.arange(tx.stop - ty.stop * sps) + psi[-1]])

    signal = signal * jnp.exp(-1j * psi_ext)[:, None]
    return signal

                
def mimoaf(
    scope: Scope,
    signal,
    taps=32,
    rtap=None,
    dims=2,
    sps=2,
    train=False,
    mimofn=af.ddlms,
    mimokwargs={},
    mimoinitargs={}):

    x, t = signal
    t = scope.variable('const', 't', conv1d_t, t, taps, rtap, 2, 'valid').value
    x = xop.frame(x, taps, sps)
    mimo_init, mimo_update, mimo_apply = mimofn(train=train, **mimokwargs)
    state = scope.variable('af_state', 'mimoaf',
                           lambda *_: (0, mimo_init(dims=dims, taps=taps, **mimoinitargs)), ())
    truth_var = scope.variable('aux_inputs', 'truth',
                               lambda *_: None, ())
    truth = truth_var.value
    if truth is not None:
        truth = truth[t.start: truth.shape[0] + t.stop]
    af_step, af_stats = state.value
    af_step, (af_stats, (af_weights, _)) = af.iterate(mimo_update, af_step, af_stats, x, truth)
    y = mimo_apply(af_weights, x)
    state.value = (af_step, af_stats)
    return Signal(y, t)
      

def channel_shuffle(x, groups):
    batch_size, channels = x.shape
    assert channels % groups == 0, "channels should be divisible by groups"
    channels_per_group = channels // groups
    x = x.reshape(batch_size, groups, channels_per_group)
    x = jnp.transpose(x, (0, 2, 1)).reshape(batch_size, -1)
    return x

from jax.nn.initializers import orthogonal
from jax.nn.initializers import orthogonal, glorot_uniform 



def squeeze_excite_attention(x):
    avg_pool = jnp.max(x, axis=0, keepdims=True)
    attention = jnp.tanh(avg_pool)
    attention = jnp.tile(attention, (x.shape[0], 1))
    x = x * attention
    return attention

def complex_channel_attention(x):
    x_real = jnp.real(x)
    x_imag = jnp.imag(x)
    x_real = squeeze_excite_attention(x_real)
    x_imag = squeeze_excite_attention(x_imag)
    x = x_real + 1j * x_imag
    return x



def generate_hippo_matrix(size):
    n = size
    P = jnp.arange(1, n+1)
    A = -2.0 * jnp.tril(jnp.ones((n, n)), -1) + jnp.diag(P)
    return A
  
from jax.nn.initializers import normal
import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros

  
# def fdbp(
#     scope: Scope,
#     signal,
#     steps=3,
#     dtaps=261,
#     ntaps=41,
#     sps=2,
#     d_init=delta,
#     n_init=gauss):
#     x, t = signal
#     dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))
#     for i in range(steps):
#         x, td = scope.child(dconv, name='DConv_%d' % i)(Signal(x, t))
#         c, t = scope.child(mimoconv1d, name='NConv_%d' % i)(Signal(jnp.abs(x)**2, td),
#                                                             taps=ntaps,
#                                                             kernel_init=n_init)
#         x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
#     return Signal(x, t)

def fdbp(scope: Scope,
         signal,
         steps: int = 3,
         dtaps: int = 261,
         ntaps: int = 41,
         sps: int = 2,
         d_init = delta,
         n_init = gauss,
         linear_mode: str = 'phase_cond',   # 'phase_cond' 或 'free'
         eps_lin: float = 0.01,
         K_lin: int = 4,
         dphi_max: float = 1.0,
         mode: str = 'valid'):
    """
    - linear_mode='phase_cond'：D 步使用物理相位 + 有界残差（推荐）
      否则使用自由核 conv1d（保持兼容）
    - mode: 'valid'（默认）或 'same'
    """
    x, t = signal

    # 入口检查
    if DEBUG_VALIDATE:
        _chk_array("Input to fDBP", x)

    # 构建 D 步
    if linear_mode == 'phase_cond':
        # 关键：共享参数（沿通道 vmap 但不复制 params）
        dconv = vmap_share_params(
            wpartial(conv1d_phase_cond, taps=dtaps, mode=mode,
                     eps=eps_lin, K=K_lin, dphi_max=dphi_max)
        )
    else:
        # 自由核 D（老路径）
        dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))

    # 逐步传播
    for i in range(steps):
        # D：线性相位卷积
        x, td = scope.child(dconv, name='DConv_%d' % i)(Signal(x, t))
        if DEBUG_VALIDATE and i < DEBUG_MAXPRINT:
            _chk_array(f"D[{i}] out", x)

        # 保护（避免后续指数溢出）
        x = _guard_complex(x, clip=1e3)

        # N：相位乘（MIMO 非线性核）
        c, t = scope.child(mimoconv1d, name='NConv_%d' % i)(
            Signal(jnp.abs(x) ** 2, td), taps=ntaps, kernel_init=n_init)

        if DEBUG_VALIDATE and i < DEBUG_MAXPRINT:
            _chk_array(f"N[{i}] phase c", c)

        # e^{j c} 逐样相位乘
        # 对齐裁剪（仅在 mode='valid' 时需要；'same' 时可以直接逐点乘）
        if mode == 'valid':
            x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
        else:  # mode == 'same'
            x = jnp.exp(1j * c) * x

        if DEBUG_VALIDATE and i < DEBUG_MAXPRINT:
            _chk_array(f"After N[{i}] mul", x)

    return Signal(x, t)


def complex_glorot_uniform(key, shape, dtype=jnp.complex64):
    # 对实部和虚部分别使用 Glorot 均匀初始化，再组合成复数
    real_init = nn.initializers.glorot_uniform()(key, shape, jnp.float32)
    imag_init = nn.initializers.glorot_uniform()(key, shape, jnp.float32)
    return real_init.astype(jnp.complex64) + 1j * imag_init.astype(jnp.complex64)

def residual_mlp(scope: Scope, signal: Signal, hidden_dim=2):
    """
    对多通道复数输入 x(t)，先做均值（或范数）处理 => 得到每个时间步一个标量，
    然后使用两层 MLP 生成 (N,) 复数 residual。
    """
    x, t = signal
    # x 的形状例如 (N, 2) 或 (N, C) 等
    # 1) 沿通道维度做均值（也可换成范数，如 jnp.linalg.norm(x, axis=-1)）
    # x_scalar = jnp.mean(x, axis=-1)  # shape=(N,), 复数
    x_scalar = jnp.linalg.norm(x, axis=-1)
    N = x_scalar.shape[0]
    # 2) reshape 成 (N,1)，并转换为复数数据类型
    x_2d = x_scalar.reshape(N, 1).astype(jnp.complex64)
    # 3) 定义 2 层 MLP 的参数，注意参数的 dtype 为 jnp.complex64
    W1 = scope.param('W1', complex_glorot_uniform, (1, hidden_dim))
    b1 = scope.param('b1',
                     lambda key, shape, dtype=jnp.complex64: jnp.zeros(shape, dtype=jnp.complex64),
                     (hidden_dim,))
    W2 = scope.param('W2', complex_glorot_uniform, (hidden_dim, 1))
    b2 = scope.param('b2',
                     lambda key, shape, dtype=jnp.complex64: jnp.zeros(shape, dtype=jnp.complex64),
                     (1,))
    # 4) 第一层全连接：hidden 的形状为 (N, hidden_dim)
    h = jnp.dot(x_2d, W1) + b1
    h = jax.nn.gelu(h)
    # 5) 输出层：形状 (N,1)
    out = jnp.dot(h, W2) + b2
    # 6) squeeze 得到形状 (N,)
    out_1d = out.squeeze(axis=-1)
    return out_1d, t
                             

def conv1d_ffn(scope: Scope, signal, taps=31, rtap=None, mode='valid', kernel_init=delta, conv_fn=xop.convolve, hidden_dim=2, use_alpha=True):
    """
    对多通道复数输入 x(t)，先做均值（或范数）处理 => 得到每个时间步一个标量，
    然后使用两层 MLP 生成 (N,) 复数 residual。
    """
    x, t = signal
    # x 的形状例如 (N, 2) 或 (N, C) 等
    # 1) 沿通道维度做均值（也可换成范数，如 jnp.linalg.norm(x, axis=-1)）
    # x_scalar = jnp.mean(x, axis=-1)  # shape=(N,), 复数
    x_scalar = jnp.linalg.norm(x, axis=-1)
    N = x_scalar.shape[0]
    # 2) reshape 成 (N,1)，并转换为复数数据类型
    x_2d = x_scalar.reshape(N, 1).astype(jnp.complex64)
    # 3) 定义 2 层 MLP 的参数，注意参数的 dtype 为 jnp.complex64
    W1 = scope.param('W1', complex_glorot_uniform, (1, hidden_dim))
    b1 = scope.param('b1',
                     lambda key, shape, dtype=jnp.complex64: jnp.zeros(shape, dtype=jnp.complex64),
                     (hidden_dim,))
    W2 = scope.param('W2', complex_glorot_uniform, (hidden_dim, 1))
    b2 = scope.param('b2',
                     lambda key, shape, dtype=jnp.complex64: jnp.zeros(shape, dtype=jnp.complex64),
                     (1,))
    # 4) 第一层全连接：hidden 的形状为 (N, hidden_dim)
    h = jnp.dot(x_2d, W1) + b1
    h = jax.nn.gelu(h)
    # 5) 输出层：形状 (N,1)
    out = jnp.dot(h, W2) + b2
    # 6) squeeze 得到形状 (N,)
    out_1d = out.squeeze(axis=-1)
    return out_1d, t  

def fdbp1(
    scope: Scope,
    signal,
    steps=3,
    dtaps=261,
    ntaps=41,
    sps=2,
    ixpm_window=3,  # IXPM 窗口大小
    d_init=delta,
    n_init=gauss):
    
    x, t = signal
    dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))
    # dconv = wpartial(dconv_pair, taps=dtaps, kinit=d_init)
    # 定义一个可训练参数 ixpm_alpha，形状为 (2*ixpm_window+1,)
    ixpm_alpha = scope.param('ixpm_alpha', nn.initializers.zeros, (2*ixpm_window+1,))
    
    for i in range(steps):
        x, td = scope.child(dconv, name='DConv1_%d' % i)(Signal(x, t))
        # 对信号幅度平方进行 roll
        ixpm_samples = [jnp.roll(jnp.abs(x)**2, shift) for shift in range(-ixpm_window, ixpm_window+1)]
        # 用 softmax 得到归一化权重
        weights = jax.nn.softmax(ixpm_alpha)
        # 计算加权和
        ixpm_power = sum(w * sample for w, sample in zip(weights, ixpm_samples))
        c, t = scope.child(mimoconv1d, name='NConv1_%d' % i)(
            Signal(ixpm_power, td), taps=ntaps, kernel_init=n_init)
        # 更新信号 x
        x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
    return Signal(x, t)

      
def identity(scope, inputs):
    return inputs


def fanout(scope, inputs, num):
    return (inputs,) * num

def fanin_sum(scope, inputs):
    val = sum(signal.val for signal in inputs)
    t = inputs[0].t  # 假设所有的 t 都相同
    return Signal(val, t)
  
def fanin_mean(scope, inputs):
    val = sum(signal.val for signal in inputs) / len(inputs)
    t = inputs[0].t  # 假设所有的 t 都相同
    return Signal(val, t)


def fanin_concat(scope, inputs, axis=-1):
    # 假设 inputs 是一个包含多个 Signal 对象的列表
    # 我们需要将每个 Signal 的 val 属性在指定轴上进行拼接
    vals = [signal.val for signal in inputs]
    concatenated_val = jnp.concatenate(vals, axis=axis)
    t = inputs[0].t  # 假设所有的 t 都相同
    return Signal(concatenated_val, t)


 
def fanin_weighted_sum(scope, inputs):
    num_inputs = len(inputs)
    print(num_inputs)
    weights = scope.param('weights', nn.initializers.ones, (num_inputs,))
    weights = jax.nn.softmax(weights)  # 归一化权重
    val = sum(w * signal.val for w, signal in zip(weights, inputs))
    t = inputs[0].t
    return Signal(val, t)
  
def fanin_attention(scope, inputs):
    num_inputs = len(inputs)
    # 初始化可训练的权重参数，形状为 (num_inputs,)
    weights = scope.param('weights', nn.initializers.zeros, (num_inputs,))
    # 使用 softmax 将权重归一化，使其之和为 1
    normalized_weights = jax.nn.softmax(weights)
    # 计算加权和
    val = sum(w * signal.val for w, signal in zip(normalized_weights, inputs))
    t = inputs[0].t  # 假设所有的 t 都相同
    return Signal(val, t)


def serial(*fs):
    def _serial(scope: Scope, inputs, **kwargs):
        for f in fs:
            if isinstance(f, tuple) or isinstance(f, list):
                name, f = f
            else:
                name = None
            inputs = scope.child(f, name=name)(inputs, **kwargs)
        return inputs
    return _serial

def parallel(*fs):
    def _parallel(scope: Scope, inputs, **kwargs):
        outputs = []
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs] * len(fs)
        for (name, f), inp in zip(fs, inputs):
            output = scope.child(f, name=name)(inp, **kwargs)
            outputs.append(output)
        return outputs
    return _parallel
