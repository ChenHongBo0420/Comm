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
  
def batchpowernorm1(scope, signal, momentum=0.999, mode='train'):
    running_mean = scope.variable('norm', 'running_mean',
                                  lambda *_: 0. + jnp.ones(signal.val.shape[-1]), ())
    if mode == 'train':
        mean = jnp.mean(jnp.abs(signal.val)**2, axis=0)
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
    conv_fn = xop.convolve):

    x, t = signal
    t = scope.variable('const', 't', conv1d_t, t, taps, rtap, 1, mode).value
    h = scope.param('kernel',
                     kernel_init,
                     (taps,), np.complex64)
    x = conv_fn(x, h, mode=mode)
    return Signal(x, t)
      
def conv1d1(
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

def mimofoeaf1(scope: Scope,
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
      
def mimoaf1(
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
#         x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0])
#     return Signal(x, t)

def channel_shuffle(x, groups):
    batch_size, channels = x.shape
    assert channels % groups == 0, "channels should be divisible by groups"
    channels_per_group = channels // groups
    x = x.reshape(batch_size, groups, channels_per_group)
    x = jnp.transpose(x, (0, 2, 1)).reshape(batch_size, -1)
    return x

from jax.nn.initializers import orthogonal
from jax.nn.initializers import orthogonal, glorot_uniform 
# ############### 

# def squeeze_excite_attention(x):
#     avg_pool = jnp.max(x, axis=0, keepdims=True)
#     attention = jnp.tanh(avg_pool)
#     attention = jnp.tile(attention, (x.shape[0], 1))
#     x = x * attention
#     return x

# def complex_channel_attention(x):
#     x_real = jnp.real(x)
#     x_imag = jnp.imag(x)
#     x_real = squeeze_excite_attention(x_real)
#     x_imag = squeeze_excite_attention(x_imag)
#     x = x_real + 1j * x_imag
#     return x
# ###############  



# class LinearRNN:
#     def __init__(self, input_dim, hidden_size, output_dim):
#         self.hidden_size = hidden_size
#         self.Wxh = orthogonal()(random.PRNGKey(0), (input_dim, hidden_size))
#         self.Whh = orthogonal()(random.PRNGKey(1), (hidden_size, hidden_size))
#         self.Why = orthogonal()(random.PRNGKey(2), (hidden_size, output_dim))
    
#     def __call__(self, x, hidden_state=None):
#         if hidden_state is None:
#             hidden_state = jnp.zeros((x.shape[0], self.hidden_size))
        
#         hidden_state = jnp.dot(x, self.Wxh) + jnp.dot(hidden_state, self.Whh)
#         output = jnp.dot(hidden_state, self.Why)
        
#         return output

def squeeze_excite_attention(x):
    avg_pool = jnp.max(x, axis=0, keepdims=True)
    attention = jnp.tanh(avg_pool)
    attention = jnp.tile(attention, (x.shape[0], 1))
    x = x * attention
    return x

def complex_channel_attention(x):
    x_real = jnp.real(x)
    x_imag = jnp.imag(x)
    x_real = squeeze_excite_attention(x_real)
    x_imag = squeeze_excite_attention(x_imag)
    x = x_real + 1j * x_imag
    return x

# class LinearRNN:
#     def __init__(self, input_dim, hidden_size, output_dim):
#         self.hidden_size = hidden_size
#         self.Wxh = orthogonal()(random.PRNGKey(0), (input_dim, hidden_size))
#         self.Whh = orthogonal()(random.PRNGKey(1), (hidden_size, hidden_size))
#         self.Why = orthogonal()(random.PRNGKey(2), (hidden_size, output_dim))
    
#     def __call__(self, x, hidden_state=None):
#         if hidden_state is None:
#             hidden_state = jnp.zeros((x.shape[0], self.hidden_size))
        
#         hidden_state = jnp.dot(x, self.Wxh) + jnp.dot(hidden_state, self.Whh)
#         output = jnp.dot(hidden_state, self.Why)
        
#         return output
      
# class TwoLayerRNN:
#     def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2

#         self.Wxh1 = orthogonal()(random.PRNGKey(0), (input_dim, hidden_size1))
#         self.Whh1 = orthogonal()(random.PRNGKey(1), (hidden_size1, hidden_size1))
#         self.Wxh2 = orthogonal()(random.PRNGKey(2), (hidden_size1, hidden_size2))
#         self.Whh2 = orthogonal()(random.PRNGKey(3), (hidden_size2, hidden_size2))

#         self.Why = orthogonal()(random.PRNGKey(4), (hidden_size2, output_dim))
    
#     def __call__(self, x, hidden_state1=None, hidden_state2=None):
#         if hidden_state1 is None:
#             hidden_state1 = jnp.zeros((x.shape[0], self.hidden_size1))
#         if hidden_state2 is None:
#             hidden_state2 = jnp.zeros((x.shape[0], self.hidden_size2))
        
#         hidden_state1 = jnp.dot(x, self.Wxh1) + jnp.dot(hidden_state1, self.Whh1)
#         hidden_state2 = jnp.dot(hidden_state1, self.Wxh2) + jnp.dot(hidden_state2, self.Whh2)
#         output = jnp.dot(hidden_state2, self.Why)
        
#         return output

# class SSM:
#     def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2

#         # 状态转移矩阵 A 和输入矩阵 B
#         self.A1 = orthogonal()(random.PRNGKey(0), (hidden_size1, hidden_size1))
#         self.B1 = orthogonal()(random.PRNGKey(1), (input_dim, hidden_size1))
#         self.A2 = orthogonal()(random.PRNGKey(2), (hidden_size2, hidden_size2))
#         self.B2 = orthogonal()(random.PRNGKey(3), (hidden_size1, hidden_size2))

#         # 观测矩阵 C
#         self.C = orthogonal()(random.PRNGKey(4), (hidden_size2, output_dim))
    
#     def __call__(self, x, hidden_state1=None, hidden_state2=None):
#         if hidden_state1 is None:
#             hidden_state1 = jnp.zeros((x.shape[0], self.hidden_size1))
#         if hidden_state2 is None:
#             hidden_state2 = jnp.zeros((x.shape[0], self.hidden_size2))
        
#         # 状态方程
#         hidden_state1 = jnp.dot(hidden_state1, self.A1) + jnp.dot(x, self.B1)
#         hidden_state2 = jnp.dot(hidden_state2, self.A2) + jnp.dot(hidden_state1, self.B2)
        
#         # 观测方程
#         output = jnp.dot(hidden_state2, self.C)
        
#         return output

def generate_hippo_matrix(size):
    n = size
    P = jnp.arange(1, n+1)
    A = -2.0 * jnp.tril(jnp.ones((n, n)), -1) + jnp.diag(P)
    return A
from jax.nn.initializers import normal
# def twolayerrnn(scope, signal, 
#                   hidden_state1=None, hidden_state2=None,
#                   input_dim=None, 
#                   hidden_size1=2, hidden_size2=2, output_dim=2):
#     """
#     两层 RNN 的前向传播函数，接口与 conv1d1 类似：
#       - scope: 用于参数和常量的初始化（如 scope.param、scope.variable）
#       - signal: 一个包含 (x, t) 的元组，此处仅使用 x 作为输入
#       - hidden_state1, hidden_state2: 可选的初始隐藏状态，若未提供则初始化为零张量
#       - input_dim: 如果没有从 signal 中获得，则需要明确指定输入维度

#     返回:
#       Signal 对象，包含输出和时间向量 t（此处 t 来自 signal 中）
#     """
#     # 从 signal 中提取输入数据 x 和时间信息 t（t 可用于调试或传递给后续模块）
#     x, t = signal

#     # 若未显式提供输入维度，则从 x 中推断（假设 x 的最后一个维度为 input_dim）
#     if input_dim is None:
#         input_dim = x.shape[-1]

#     # 初始化 HIPPO 状态转移矩阵作为常量（使用 scope.variable）
#     A1 = scope.variable('const', 'A1', generate_hippo_matrix, hidden_size1).value
#     A2 = scope.variable('const', 'A2', generate_hippo_matrix, hidden_size2).value

#     # 使用 scope.param 初始化可训练参数矩阵
#     B1 = scope.param('B1', orthogonal(), (input_dim, hidden_size1), jnp.float32)
#     B2 = scope.param('B2', orthogonal(), (hidden_size1, hidden_size2), jnp.float32)
#     C  = scope.param('C', orthogonal(), (hidden_size2, output_dim), jnp.float32)

#     # 初始化隐藏状态（若未提供，则使用零张量，batch_size 从 x 的第一维推断）
#     batch_size = x.shape[0]
#     if hidden_state1 is None:
#         hidden_state1 = jnp.zeros((batch_size, hidden_size1))
#     if hidden_state2 is None:
#         hidden_state2 = jnp.zeros((batch_size, hidden_size2))

#     # 第一层 RNN 状态更新与注意力机制
#     hidden_state1 = jnp.dot(hidden_state1, A1) + jnp.dot(x, B1)
#     # hidden_state1 = squeeze_excite_attention(hidden_state1)

#     # 第二层 RNN 状态更新与注意力机制
#     hidden_state2 = jnp.dot(hidden_state2, A2) + jnp.dot(hidden_state1, B2)
#     # hidden_state2 = complex_channel_attention(hidden_state2)

#     # 输出：使用观测矩阵 C 得到最终输出
#     output = jnp.dot(hidden_state2, C)
#     return Signal(output, t)

from jax.nn.initializers import orthogonal, zeros

import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros

import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros



                   
# class ThreeLayerRNN_SSM:
#     def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2
  

#         # 使用 HIPPO 矩阵初始化状态转移矩阵 A
#         self.A1 = generate_hippo_matrix(hidden_size1)
#         self.A2 = generate_hippo_matrix(hidden_size2)
#         self.A3 = generate_hippo_matrix(hidden_size2)
        
#         # 输入矩阵 B
#         self.B1 = orthogonal()(random.PRNGKey(1), (input_dim, hidden_size1))
#         self.B2 = orthogonal()(random.PRNGKey(2), (hidden_size1, hidden_size2))
#         self.B3 = orthogonal()(random.PRNGKey(3), (hidden_size2, hidden_size2))

#         # 观测矩阵 C
#         self.C = orthogonal()(random.PRNGKey(4), (hidden_size2, output_dim))
    
#     def __call__(self, x, hidden_state1=None, hidden_state2=None):
      
#         if hidden_state1 is None:
#             hidden_state1 = jnp.zeros((x.shape[0], self.hidden_size1))
#         if hidden_state2 is None:
#             hidden_state2 = jnp.zeros((x.shape[0], self.hidden_size2))
        
#         # 第一层状态更新
#         hidden_state1 = jnp.dot(hidden_state1, self.A1) + jnp.dot(x, self.B1)
        
#         hidden_state2 = jnp.dot(hidden_state2, self.A2) + jnp.dot(hidden_state1, self.B2)

#         hidden_state3 = jnp.dot(hidden_state2, self.A3) + jnp.dot(hidden_state2, self.B3)
        
#         output = jnp.dot(hidden_state2, self.C)
  
#         return output

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = orthogonal()(random.PRNGKey(0), (input_dim, output_dim))
        
    def __call__(self, x):
        return jnp.dot(x, self.W) 

def weighted_interaction(x1, x2):
    # 定义简单的加权相互作用
    weight = jnp.mean(x1 * x2)  # 计算交互权重
    x1_updated = x1 + weight * x2  # 加权求和
    x2_updated = x2 + weight * x1  # 加权求和
    return x1_updated, x2_updated

# def weighted_interaction(x1, x2):
#     x1_normalized = (x1 - jnp.mean(x1)) / (jnp.std(x1) + 1e-6)
#     x2_normalized = (x2 - jnp.mean(x2)) / (jnp.std(x2) + 1e-6)
#     weight = jnp.mean(x1_normalized * x2_normalized)
#     x1_updated = x1 + weight * x2
#     x2_updated = x2 + weight * x1
#     return x1_updated, x2_updated    
  
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

# def fdbp(
#     scope: Scope,
#     signal,
#     steps=3,
#     dtaps=261,
#     ntaps=41,
#     sps=2,
#     d_init=delta,
#     n_init=gauss,
#     mu=0.0001  # 学习率，用于 LMS 更新 gamma，需根据实际情况调参
# ):
#     """
#     自适应 DBP：在每一步补偿中对相位补偿的缩放因子 gamma 进行自适应更新，
#     以减少误差累积，提升整体补偿性能。
#     """
#     x, t = signal
#     # 初始化自适应相位缩放因子 gamma
#     gamma = 1.0
#     # 构造局部时域卷积函数（通过 vmap 包裹 conv1d）
#     dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))
    
#     for i in range(steps):
#         # 执行局部色散补偿步骤
#         x, td = scope.child(dconv, name=f'DConv_{i}')(Signal(x, t))
#         # 执行非线性补偿步骤：计算相位校正 c
#         c, t = scope.child(mimoconv1d, name=f'NConv_{i}')(
#             Signal(jnp.abs(x)**2, td), taps=ntaps, kernel_init=n_init)
#         # 应用自适应相位补偿：使用 gamma 对 c 进行缩放
#         x_new = jnp.exp(1j * gamma * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
        
#         # 计算误差：例如用当前步骤补偿前后的平均功率差异作为误差信号
#         # 这里的 error 定义可以根据实际需求进行设计
#         power_before = jnp.mean(jnp.square(jnp.abs(x)))
#         power_after = jnp.mean(jnp.square(jnp.abs(x_new)))
#         error = power_after - power_before
        
#         # 更新 gamma（LMS 规则）：gamma_new = gamma - mu * error
#         # 注意：根据实际误差定义，可能需要调整符号
#         gamma = gamma - mu * error
        
#         # 将更新后的信号作为下一步输入
#         x = x_new
#     return Signal(x, t)

def improved_dispersion_compensator(x, dt, beta2, beta3=0.0, beta4=0.0):
    """
    利用 FFT/IFFT 在频域中施加色散补偿，支持二阶、三阶和四阶色散。
    
    :param x: 输入时域信号（1D 复数数组）
    :param dt: 采样时间间隔
    :param beta2: 二阶色散系数
    :param beta3: 三阶色散系数（默认为 0.0）
    :param beta4: 四阶色散系数（默认为 0.0）
    :return: 补偿后的时域信号
    """
    X = jnp.fft.fft(x)
    N = x.shape[-1]
    freqs = jnp.fft.fftfreq(N, d=dt)
    omega = 2 * jnp.pi * freqs

    # 计算色散补偿相位（负号表示反向补偿）
    phase = -0.5 * beta2 * (omega ** 2)
    if beta3 != 0.0:
        phase += (1.0 / 6.0) * beta3 * (omega ** 3)
    if beta4 != 0.0:
        phase += - (1.0 / 24.0) * beta4 * (omega ** 4)

    X_comp = X * jnp.exp(1j * phase)
    x_comp = jnp.fft.ifft(X_comp)
    return x_comp

def improved_dconv(signal: Signal, beta2, beta3, beta4):
    """
    改进版色散补偿模块，利用 FFT/IFFT 实现高阶色散补偿，
    返回补偿后的信号和时间轴（保持不变）。
    """
    x, t = signal
    dt = t[1] - t[0]  # 假设 t 为均匀采样数组
    x_disp = improved_dispersion_compensator(x, dt, beta2, beta3, beta4)
    # 返回与原来接口一致的 tuple
    return x_disp, t
  
def fdbp(scope, signal, steps=3,
         dtaps=261, ntaps=41, sps=2,
         d_init=None,    # 保留原接口（本例中未使用）
         n_init=None,    # 保留原接口（本例中未使用）
         beta2=1.0, beta3=0.0, beta4=0.0):
    """
    改进版 DBP 算法：仅替换色散补偿部分，采用 improved_dconv 进行 FFT-based 色散补偿，
    非线性补偿部分保持不变，调用原有 mimoconv1d 模块。

    :param scope: 参数作用域对象，用于管理子模块
    :param signal: 输入 Signal 对象（包含 x, t）
    :param steps: 迭代步数
    :param dtaps: 原色散补偿卷积核 tap 数（预留接口）
    :param ntaps: 非线性补偿卷积核 tap 数（预留接口）
    :param sps: 每符号采样数（预留接口）
    :param d_init: 色散补偿核初始化函数（预留接口）
    :param n_init: 非线性补偿核初始化函数（预留接口）
    :param beta2: 二阶色散系数
    :param beta3: 三阶色散系数
    :param beta4: 四阶色散系数
    :return: 补偿后的 Signal 对象
    """
    x, t = signal

    for i in range(steps):
        # 用 improved_dconv 替换原有的 dconv（色散补偿）
        dispersion_fn = lambda sc, sig: improved_dconv(sig, beta2, beta3, beta4)
        x, td = scope.child(dispersion_fn, name=f'DConv_{i}')(Signal(x, t))
        
        # 非线性补偿部分保持不变
        c, t_new = scope.child(mimoconv1d, name=f'NConv_{i}')(
            Signal(jnp.abs(x)**2, td),
            taps=ntaps,
            kernel_init=n_init
        )
        # 按照原有公式更新 x，时间轴切片操作保持不变
        x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
        # 可选择更新 t，这里直接用 t_new（或保持 td，根据具体实现）
        t = t_new

    return Signal(x, t)
           
def fdbp1(
    scope: Scope,
    signal,
    steps=3,
    dtaps=261,
    ntaps=41,
    sps=2,
    ixpm_window=7,  # 新增参数，设置IXPM的窗口大小
    d_init=delta,
    n_init=gauss):
    
    x, t = signal
    dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))
    
    # input_dim = x.shape[1]
    # hidden_size = 2 
    # output_dim = x.shape[1]
    # x1 = x[:, 0]
    # x2 = x[:, 1]
    # x1_updated, x2_updated = weighted_interaction(x1, x2)
    # x_updated = jnp.stack([x1_updated, x2_updated], axis=1)
    # rnn_layer = TwoLayerRNN(input_dim, hidden_size, hidden_size, output_dim)
    # x = rnn_layer(x_updated)
    for i in range(steps):
        x, td = scope.child(dconv, name='DConv_%d' % i)(Signal(x, t))
        ixpm_samples = [
            jnp.roll(jnp.abs(x)**2, shift) for shift in range(-ixpm_window, ixpm_window + 1)
        ]
        ixpm_power = sum(ixpm_samples) / (2 * ixpm_window + 1)
        c, t = scope.child(mimoconv1d, name='NConv_%d' % i)(Signal(ixpm_power, td),
                                                            taps=ntaps,
                                                            kernel_init=n_init)
        x = jnp.exp(1j * c) * x[t.start - td.start: t.stop - td.stop + x.shape[0]]
    return Signal(x, t)

      
# def fdbp1(
#     scope: Scope,
#     signal,
#     steps: int = 3,
#     dtaps: int = 261,
#     ntaps: int = 41,
#     sps: int = 2,
#     max_ixpm_window: int = 5,  # 允许网络在 [-max_ixpm_window, +max_ixpm_window] 内学习
#     d_init = delta,
#     n_init = gauss
# ):
#     """
#     与原fdbp1相似，但IXPM窗口在 [-max_ixpm_window, +max_ixpm_window]范围内
#     引入一个可学习的softmax权重(2*max_ixpm_window+1个参数)，
#     让网络决定每个shift的重要度，而不是均匀平均。
#     """
#     x, t = signal

#     # --- 1) 定义色散滤波 dconv (和你原先一样) ---
#     dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))

#     # --- 2) 声明一个可学习向量 alpha，用 softmax 得到 weight ---
#     #    alpha.shape = (2*max_ixpm_window + 1,)
#     alpha = scope.param(
#         'ixpm_alpha',
#         lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),  # init 0 or normal
#         (2 * max_ixpm_window + 1,)
#     )
#     # 计算 softmax, 保证非负且 sum=1
#     alpha_stable = alpha - jnp.max(alpha)  # 防止溢出
#     w = jnp.exp(alpha_stable)
#     w = w / jnp.sum(w)

#     # 主循环 steps
#     for i in range(steps):
#         # 2.1) 线性色散
#         x, td = scope.child(dconv, name=f'DConv_{i}')(Signal(x, t))

#         # 2.2) 计算 IXPM: sum over shifts(roll), 并由 w 加权
#         x_abs2 = jnp.abs(x)**2
#         ixpm_power = jnp.zeros_like(x_abs2)  # same shape as x_abs2
#         # 对 shift in [-max_ixpm_window, ..., +max_ixpm_window] 做 roll
#         for idx, shift in enumerate(range(-max_ixpm_window, max_ixpm_window + 1)):
#             ixpm_power += w[idx] * jnp.roll(x_abs2, shift, axis=0)

#         # 2.3) 做 mimoconv1d => c
#         c, t_new = scope.child(mimoconv1d, name=f'NConv_{i}')(Signal(ixpm_power, td),
#                                                               taps=ntaps,
#                                                               kernel_init=n_init)
#         # 2.4) 相位补偿
#         #    这里与原版相同: x = exp(1j * c) * x[对齐切片]
#         x_slice = x[t_new.start - td.start : t_new.stop - td.stop + x.shape[0]]
#         x = jnp.exp(1j * c) * x_slice
#         t = t_new  # 更新时间戳

#     return Signal(x, t)


def identity(scope, inputs):
    return inputs


# def fanout(scope, inputs, num):
#     return (inputs,) * num
      
# # compositors

# def serial(*fs):
#     def _serial(scope, inputs, **kwargs):
#         for f in fs:
#             if isinstance(f, tuple) or isinstance(f, list):
#                 name, f = f
#             else:
#                 name = None
#             inputs = scope.child(f, name=name)(inputs, **kwargs)
#         return inputs
#     return _serial


# def parallel(*fs):
#     def _parallel(scope, inputs, **kwargs):
#         outputs = []
#         for f, inp in zip(fs, inputs):
#             if isinstance(f, tuple) or isinstance(f, list):
#                 name, f = f
#             else:
#                 name = None
#             out = scope.child(f, name=name)(inp, **kwargs)
#             outputs.append(out)
#         return outputs
#     return _parallel

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
