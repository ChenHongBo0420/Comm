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
        tslice = (-delay * stride, taps - stride * (rtap + 1))  # TODO: think more about this
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
    vf.__name__ = 'vmapped_' + f.__name__  # [Workaround]: lifted transformation does not keep the original name
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
        mean = jnp.mean(jnp.abs(signal.val) ** 2, axis=0)
        running_mean.value = momentum * running_mean.value + (1 - momentum) * mean
    else:
        mean = running_mean.value
    return signal / jnp.sqrt(mean)


def conv1d(
        scope: Scope,
        signal,
        taps=31,
        rtap=None,
        mode='valid',
        kernel_init=delta,
        conv_fn=xop.convolve):
    x, t = signal
    t = scope.variable('const', 't', conv1d_t, t, taps, rtap, 1, mode).value
    h = scope.param('kernel',
                    kernel_init,
                    (taps,), np.complex64)
    x = conv_fn(x, h, mode=mode)
    return Signal(x, t)


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
    y, ty = auxsig  # assume y is continuous in time
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
    P = jnp.arange(1, n + 1)
    A = -2.0 * jnp.tril(jnp.ones((n, n)), -1) + jnp.diag(P)
    return A


from jax.nn.initializers import normal
import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros, glorot_uniform

# ------------------------------------------------------------
# Time‑domain 1‑D CNN  （等价于 vmap(conv1d) + GELU）
# ------------------------------------------------------------
def TimeCNN(scope, signal, taps=61, hidden=2,
            k_init=delta,                # <<< 替换成 glorot_uniform if you like
            act=jax.nn.gelu, name='TimeCNN'):
    x, t = signal                       # x:(N,C)
    C_in  = x.shape[-1]
    k = scope.param('kernel', k_init,      # (taps,Cin,Cout)
                    (taps, C_in, hidden), np.float32)
    y = xcomm.mimoconv(x, k, mode='same')  # SAME 长度不变
    y = act(y)
    return Signal(y, t)

# ------------------------------------------------------------
# 轻量 Gated RNN  （单步循环，保持时长）
# ------------------------------------------------------------
import commplax.module.core as cxcore
from commplax.module import core as ccore    

def GatedRNN(scope, signal, hidden=2, hippo=False, name='GatedRNN'):
    x, t = signal
    H, dtype = hidden, x.dtype
    init_h = jnp.zeros((x.shape[0], H), dtype=dtype)

    def w(n, shape):
        return scope.param(n, glorot_uniform(), shape, dtype).astype(dtype)

    if hippo:
        A = generate_hippo_matrix(H).astype(dtype)

    def step(h, x_t):
        z = jax.nn.sigmoid(h @ w('Wz', (H, H)))
        r = jax.nn.sigmoid(h @ w('Wr', (H, H)))
        h_tilde = jnp.tanh(x_t @ w('Wx', (x.shape[-1], H)) +
                           (r * h) @ w('Wh', (H, H)))
        h_next = (1 - z) * h + z * h_tilde
        if hippo:
            h_next = h_next + h_next @ A
        return h_next, h_next

    _, h_seq = jax.lax.scan(step, init_h, x)      # 不再 swap
    y = h_seq
    return ccore.Signal(y, t) 



# ------------------------------------------------------------
# Dense head  -> I/Q (或你想要的维度)
# ------------------------------------------------------------
def DenseHead(scope, signal, out_dim=2, name='DenseHead'):
    x, t = signal                       # x:(N,H)
    if x.ndim == 3:                     # 有时间维时逐符号映射
        N,T,H = x.shape
        x = x.reshape(-1, H)
    H = x.shape[-1]
    W = scope.param('W', orthogonal(), (H, out_dim))
    b = scope.param('b', zeros, (out_dim,))
    y = x @ W + b
    if 'T' in locals():                 # restore time dim
        y = y.reshape(N, T, out_dim)
    return Signal(y, t)


def fdbp(scope: Scope,
         signal,
         *,
         steps      = 3,
         dtaps      = 261,
         ntaps      = 41,      # 保留形参兼容外层
         sps        = 2,
         d_init = delta,
         n_init = gauss,       # ← 不再使用，但保持签名
         hidden_dim = 32):     # 可在 make_base_module 调整
    """
    ‑ 与旧版接口/返回完全一致；
    ‑ 内部做：一次 D‑Conv  ➜  轻量复值 Gated‑RNN  ➜  回 2‑pol complex；
    ‑ 输出 shape = (N, 2) complex64，可直接送后续 MIMO‑AF。
    """
    # ===== 1. Dispersion compensation (SAME) ====================
    x_cplx, t_sig = signal                       # x:(N,2) complex
    dconv = vmap(wpartial(conv1d,
                          taps=dtaps,
                          kernel_init=d_init))
    x_cplx, t = scope.child(dconv, 'DConv')(Signal(x_cplx, t_sig))

    # ===== 2. 复→实向量  (N,4)  [ReX, ImX, ReY, ImY] ============
    x4 = jnp.concatenate([jnp.real(x_cplx),
                          jnp.imag(x_cplx)], axis=-1)

    # ===== 3. 单层 Gated‑RNN  ===================================
    H = hidden_dim
    dtype = x4.dtype

    # 参数
    Wi = scope.param('Wi', glorot_uniform(), (4, 3*H), dtype)  # input→[z,r,ĥ]
    Wh = scope.param('Wh', glorot_uniform(), (H, 3*H), dtype)  # hidden→…
    b  = scope.param('b',  zeros,           (3*H,),  dtype)

    def gru_cell(h, x_t):
        z_r_h = x_t @ Wi + h @ Wh + b         # (3H,)
        z, r, h_hat = jnp.split(z_r_h, 3)
        z = jax.nn.sigmoid(z)
        r = jax.nn.sigmoid(r)
        h_hat = jax.nn.gelu(h_hat)
        h_new = (1 - z) * h + z * h_hat
        return h_new, h_new                   # carry, y

    h0 = jnp.zeros((H,), dtype)               # (H,)
    _, h_seq = jax.lax.scan(gru_cell, h0, x4) # (N,H)

    # ===== 4. 全连接回 2‑pol complex  ===========================
    W_out = scope.param('W_out', glorot_uniform(), (H, 4), dtype)
    y4 = h_seq @ W_out                        # (N,4)
    re, im = jnp.split(y4, 2, axis=-1)        # 各 (N,2)
    x_out = (re + 1j*im).astype(jnp.complex64)

    return Signal(x_out, t)


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



def conv1d_ffn(scope: Scope, signal, taps=31, rtap=None, mode='valid', kernel_init=delta, conv_fn=xop.convolve,
               hidden_dim=2, use_alpha=True):
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
    ixpm_alpha = scope.param('ixpm_alpha', nn.initializers.zeros, (2 * ixpm_window + 1,))

    for i in range(steps):
        x, td = scope.child(dconv, name='DConv1_%d' % i)(Signal(x, t))
        # 对信号幅度平方进行 roll
        ixpm_samples = [jnp.roll(jnp.abs(x) ** 2, shift) for shift in range(-ixpm_window, ixpm_window + 1)]
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
