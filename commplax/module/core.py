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
      
      
# def mimofoeaf(scope: Scope,
#               signal,
#               framesize=100,
#               w0=0,
#               train=False,
#               preslicer=lambda x: x,
#               foekwargs={},
#               mimofn=af.rde,
#               mimokwargs={},
#               mimoinitargs={}):

#     sps = 2
#     dims = 2
#     tx = signal.t
#     # MIMO
#     slisig = preslicer(signal)
#     auxsig = scope.child(mimoaf,
#                          mimofn=mimofn,
#                          train=train,
#                          mimokwargs=mimokwargs,
#                          mimoinitargs=mimoinitargs,
#                          name='MIMO4FOE')(slisig)
#     y, ty = auxsig # assume y is continuous in time
#     yf = xop.frame(y, framesize, framesize)

#     foe_init, foe_update, _ = af.array(af.frame_cpr_kf, dims)(**foekwargs)
#     state = scope.variable('af_state', 'framefoeaf',
#                            lambda *_: (0., 0, foe_init(w0)), ())
#     phi, af_step, af_stats = state.value

#     af_step, (af_stats, (wf, _)) = af.iterate(foe_update, af_step, af_stats, yf)
#     wp = wf.reshape((-1, dims)).mean(axis=-1)
#     w = jnp.interp(jnp.arange(y.shape[0] * sps) / sps,
#                    jnp.arange(wp.shape[0]) * framesize + (framesize - 1) / 2, wp) / sps
#     psi = phi + jnp.cumsum(w)
#     state.value = (psi[-1], af_step, af_stats)

#     # apply FOE to original input signal via linear extrapolation
#     psi_ext = jnp.concatenate([w[0] * jnp.arange(tx.start - ty.start * sps, 0) + phi,
#                                psi,
#                                w[-1] * jnp.arange(tx.stop - ty.stop * sps) + psi[-1]])

#     signal = signal * jnp.exp(-1j * psi_ext)[:, None]
#     return signal

def mimofoeaf(
    scope: Scope,
    signal: Signal,
    framesize=100,
    w0=0.0,             # 保持原先的 w0 接口
    train=False,
    preslicer=lambda x: x,
    foekwargs={},       # 给 frame_cpr_kf(...) 的额外参数
    mimofn=af.rde,      # 给 mimoaf 的自适应滤波函数
    mimokwargs={},
    mimoinitargs={},
    learn_R=False,      # 是否让噪声协方差 R 成为可学习参数
    learn_Q=False       # 是否让噪声协方差 Q 成为可学习参数
):
    """
    改进版: 不再调用 af.array(...) => 无 vmap 轴冲突
    """
    sps = 2
    dims = 2
    tx = signal.t

    # (1) 先做 MIMO 自适应滤波
    slisig = preslicer(signal)
    auxsig = scope.child(mimoaf,
                         mimofn=mimofn,
                         train=train,
                         mimokwargs=mimokwargs,
                         mimoinitargs=mimoinitargs,
                         name='MIMO4FOE')(slisig)
    y, ty = auxsig

    # (2) 分帧 => (N_frames, framesize, dims)
    yf = xop.frame(y, framesize, framesize)

    # (3) 直接拿 frame_cpr_kf 的 (init, update, apply), 无 array(...) 包装
    foe_init, foe_update_orig, foe_apply = af.frame_cpr_kf(**foekwargs)

    # (4) 定义可学习 R, Q(可选) -- 只是在本函数param中
    if learn_R:
        R = scope.param(
            'R',
            lambda key, shape, dtype: jnp.eye(dims, dtype=dtype)*0.01,
            (dims, dims), jnp.float32
        )
    else:
        R = jnp.eye(dims, dtype=jnp.float32)*0.01

    if learn_Q:
        Q = scope.param(
            'Q',
            lambda key, shape, dtype: jnp.eye(dims, dtype=dtype)*1e-4,
            (dims, dims), jnp.float32
        )
    else:
        Q = jnp.eye(dims, dtype=jnp.float32)*1e-4

    # (5) foe_init(w0) => 初始化
    def init_kalman():
        return foe_init(w0)  # => (z0, P0, Q, R)
    state_var = scope.variable('af_state', 'framefoeaf',
                               lambda *_: (0.0, 0, init_kalman()), ())
    phi, af_step, af_stats = state_var.value
    # af_stats=(z_c, P_c, Q, R)

    # (6) 定义 wrapper => 3参: (step_i, old_state, data_tuple)
    # 并调用 foe_update_orig(step_i, old_state, data_tuple)
    def foe_update_with_params(step_i, old_state, data_tuple):
        frame_data = data_tuple[0]  # (framesize, dims)
        # 你若有 truth: truth_data = data_tuple[1]
        # 这里不显式传 R, Q, w0
        new_state, w_frame = foe_update_orig(step_i, old_state, (frame_data,))
        # w_frame shape=(?, dims)? 你可自定义
        return new_state, (w_frame, None)

    # (7) 用 af.iterate(...) => 逐帧扫描
    af_step, (af_stats, (wf, _)) = af.iterate(
        foe_update_with_params,
        af_step,
        af_stats,
        yf  # shape=(N_frames, framesize, dims)
    )
    # wf shape=(N_frames, ???)

    # (8) 后处理 => interpolation, etc
    # 这里假设 wf=(N_frames, dims)
    # 如果 foe_update_orig 产出 shape=(N_frames, 2)  => ok
    # 你可自行调试 shape
    wp = wf.reshape((-1, dims)).mean(axis=-1)  # => (N_frames,)
    x_wp = jnp.arange(wp.shape[0]) * framesize + (framesize - 1) / 2
    x_axis = jnp.arange(y.shape[0] * sps) / sps
    w = jnp.interp(x_axis, x_wp, wp) / sps
    psi = phi + jnp.cumsum(w)
    state_var.value = (psi[-1], af_step, af_stats)

    # (9) 相位校正
    psi_ext = jnp.concatenate([
        w[0] * jnp.arange(tx.start - ty.start * sps, 0) + phi,
        psi,
        w[-1] * jnp.arange(tx.stop - ty.stop * sps) + psi[-1]
    ])
    out_signal = signal * jnp.exp(-1j * psi_ext)[:, None]
    return out_signal

# def mimofoeaf(
#     scope: Scope,
#     signal: Signal,
#     framesize=100,
#     w0=0.0,
#     train=False,
#     preslicer=lambda x: x,
#     foekwargs={},
#     mimofn=af.rde,
#     mimokwargs={},
#     mimoinitargs={},
# ):
#     sps = 2
#     dims = 2
#     x, t = signal

#     # (1) MIMO self-adaptive
#     slisig = preslicer(signal)
#     out_sig = scope.child(mimoaf,
#                           mimofn=mimofn,
#                           train=train,
#                           mimokwargs=mimokwargs,
#                           mimoinitargs=mimoinitargs,
#                           name='MIMO4FOE')(slisig)
#     y, ty = out_sig

#     # (2) 分帧 => shape=(N_frames, framesize, dims)
#     yf = xop.frame(y, framesize, framesize)

#     # (3) 直接拿 frame_cpr_kf
#     foe_init, foe_update, foe_apply = af.frame_cpr_kf(**foekwargs)

#     # (4) Kalman state => scope.variable
#     def init_kalman():
#         return foe_init(w0)
#     state_var = scope.variable('af_state', 'framefoeaf',
#                                lambda *_: (0., 0, init_kalman()), ())
#     phi, af_step, af_stats = state_var.value

#     # (5) 定义 wrapper
#     def foe_update_with_params(step_i, old_state, data_tuple):
#         frame_data = data_tuple[0]  # shape(framesize,dims)
#         new_state, w_frame = foe_update(step_i, old_state, (frame_data,))
#         return new_state, (w_frame, None)

#     # (6) af.iterate => 逐帧
#     af_step, (af_stats, (wf, _)) = af.iterate(
#         foe_update_with_params,
#         af_step,
#         af_stats,
#         yf
#     )
#     # wf shape => (N_frames,1,2) in this example

#     # (7) 后处理 => interpolation
#     #   suppose wf reshape => (N_frames, dims)
#     wf_reshaped = wf.reshape((-1, dims))  # => (N_frames,2)
#     wp = wf_reshaped.mean(axis=-1)       # => (N_frames,)
#     x_wp = jnp.arange(wp.shape[0]) * framesize + (framesize - 1)/2
#     x_axis = jnp.arange(y.shape[0]*sps)/sps
#     w_phase = jnp.interp(x_axis, x_wp, wp)/sps
#     psi = phi + jnp.cumsum(w_phase)
#     state_var.value = (psi[-1], af_step, af_stats)

#     # (8) apply phase
#     # times => shape(len(psi),), x => shape(len(psi), 2)
#     # expand psi => shape(len(psi),1)
#     psi_ext = psi[:, None]
#     out_val = x[:psi.shape[0]] * jnp.exp(-1j * psi_ext)  # or do bounds carefully
#     return Signal(out_val, t)
                
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
    return attention

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

def generate_hippo_matrix(size):
    n = size
    P = jnp.arange(1, n+1)
    A = -2.0 * jnp.tril(jnp.ones((n, n)), -1) + jnp.diag(P)
    return A
  
from jax.nn.initializers import normal
import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros


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
                             

from jax import debug
def fdbp(
    scope: Scope,
    signal,
    steps=3,
    dtaps=261,
    ntaps=41,
    sps=2,
    d_init=delta,
    n_init=gauss,
    hidden_dim=2,
    use_alpha=True,
):
    """
    保持原 fdbp(D->N)结构:
      1) D
      2) N
      + 3) residual MLP => out shape=(N,) and add to x
    """
    x, t = signal
    # 1) 色散
    dconv = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))

    # 可选: 对res加个可训练缩放
    if use_alpha:
        alpha = scope.param('res_alpha', nn.initializers.zeros, ())
    else:
        alpha = 1.0
    # debug.print("alpha = {}", alpha)
    for i in range(steps):
        # --- (A) 色散补偿 (D)
        x, td = scope.child(dconv, name='DConv_%d' % i)(Signal(x, t))
        
        # --- (B) 非线性补偿 (N)
        c, tN = scope.child(mimoconv1d, name='NConv_%d' % i)(
            Signal(jnp.abs(x)**2, td),
            taps=ntaps,
            kernel_init=n_init
        )
        # 应用相位: x_new = exp(j*c) * x[...]
        x_new = jnp.exp(1j * c) * x[tN.start - td.start : x.shape[0] + (tN.stop - td.stop)]
        # --- (C) residual MLP
        #  对 |x_new|^2 做 MLP => residual => shape=(N_new,)
        res_val, t_res = scope.child(residual_mlp, name=f'ResCNN_{i}')(
            Signal(jnp.abs(x_new)**2, tN),
            hidden_dim=hidden_dim
        )
        # res_val => (N_new,)
        # cast to complex, or interpret as real
        # 这里示例 "在幅度上+res"
        # x_new += alpha * res_val
        # 不分real/imag => 全部 real offset => x_new + alpha * res
        # 只要 x_new是complex => convert
        res_val_cplx = jnp.asarray(res_val, x_new.dtype)
        res_val_cplx_2d = res_val_cplx[:, None]    # shape (N,1)
        x_new = x_new + alpha * res_val_cplx_2d 
        
        # update x,t
        x, t = x_new, t_res
    return Signal(x, t)

# def fdbp(
#     scope: Scope,
#     signal: Signal,
#     steps: int = 3,
#     dtaps: int = 261,
#     ntaps: int = 41,
#     sps: int = 2,
#     ixpm_window: int = 7,           # 对 fdbp1 中 ixpm 的窗口大小
#     d_init=delta,
#     n_init=gauss,
#     name='fdbp2branches'
# ):
#     """
#     同时执行:
#       - fdbp: 忽略IXPM的补偿方式
#       - fdbp1: 带IXPM补偿
#     并在每一步(step)后做一次融合 (bridge)，再进入下一步。

#     Args:
#       signal:  输入信号 (Signal(val, t))
#       steps :  总共迭代多少次(与原 fdbp/fdbp1 类似)
#       dtaps :  色散滤波器长度
#       ntaps :  非线性滤波器长度
#       ixpm_window: 在 fdbp1 中计算IXPM时的 roll 窗口
#       ... 其余参见 fdbp / fdbp1 说明

#     Returns:
#       最终补偿后的 Signal。
#     """
#     x_in, t_in = signal

#     # 分别为“忽略IXPM的fdbp”和“考虑IXPM的fdbp1”各自准备一套卷积
#     # (可以共用也可以分开，这里演示显式分开以示区分)
#     dconv_ignore = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))
#     dconv_ixpm   = vmap(wpartial(conv1d, taps=dtaps, kernel_init=d_init))

#     # 用于 IXPM 加权的可训练参数
#     ixpm_alpha = scope.param('ixpm_alpha', nn.initializers.zeros, (2*ixpm_window + 1,))

#     # 融合时的可训练权重 => 每步都使用同一个 alpha (示例)
#     # 如果想每步不同，可以定义 alpha_list = scope.param(... shape=(steps,) ) 再逐步取索引
#     bridge_alpha = scope.param('bridge_alpha', nn.initializers.zeros, ())

#     # 初始化两条分支的输入都为同一个 x_in
#     x_ignore = x_in
#     t_ignore = t_in

#     x_ixpm   = x_in
#     t_ixpm   = t_in

#     for i in range(steps):
#         # --- (1) 忽略IXPM的单步 fdbp ---
#         x_ignore, t_ignore_new = scope.child(dconv_ignore, name=f"DConv_ignore_{i}")(
#             Signal(x_ignore, t_ignore)
#         )
#         # 做非线性补偿
#         c_ignore, tN_ignore = scope.child(mimoconv1d, name=f"NConv_ignore_{i}")(
#             Signal(jnp.abs(x_ignore)**2, t_ignore_new),
#             taps=ntaps, kernel_init=n_init
#         )
#         x_ignore_new = jnp.exp(1j * c_ignore) * x_ignore[
#             tN_ignore.start - t_ignore_new.start : x_ignore.shape[0] + (tN_ignore.stop - t_ignore_new.stop)
#         ]

#         # --- (2) 带IXPM的单步 fdbp1 ---
#         # 同样先做色散补偿
#         x_ixpm, t_ixpm_new = scope.child(dconv_ixpm, name=f"DConv_ixpm_{i}")(
#             Signal(x_ixpm, t_ixpm)
#         )
#         # 计算IXPM加权和
#         # 先 roll abs(x_ixpm)^2
#         ixpm_samples = [jnp.roll(jnp.abs(x_ixpm)**2, shift) for shift in range(-ixpm_window, ixpm_window+1)]
#         weights = jax.nn.softmax(ixpm_alpha)
#         ixpm_power = sum(w * sample for w, sample in zip(weights, ixpm_samples))

#         # 做非线性补偿
#         c_ixpm, tN_ixpm = scope.child(mimoconv1d, name=f"NConv_ixpm_{i}")(
#             Signal(ixpm_power, t_ixpm_new),
#             taps=ntaps, kernel_init=n_init
#         )
#         x_ixpm_new = jnp.exp(1j * c_ixpm) * x_ixpm[
#             tN_ixpm.start - t_ixpm_new.start : x_ixpm.shape[0] + (tN_ixpm.stop - t_ixpm_new.stop)
#         ]

#         # 更新各自分支的 (x, t)
#         x_ignore, t_ignore = x_ignore_new, tN_ignore
#         x_ixpm,   t_ixpm   = x_ixpm_new,   tN_ixpm

#         # --- (3) 两路结果做可训练的加权融合 => 同步更新成同一个 x_fused ---
#         # 假设要求它们下一步起就“保持一致”，则把 x_ignore / x_ixpm 都赋值为融合结果
#         x_fused = 0.4 * x_ignore + 0.6 * x_ixpm
#         t_fused = t_ignore  # = t_ixpm, 假设它们相同

#         # 同步更新 => 下一步大家都从 x_fused, t_fused 开始
#         x_ignore, t_ignore = x_fused, t_fused
#         x_ixpm,   t_ixpm   = x_fused, t_fused

#     # 最终的输出就是融合后的 x_fused
#     return Signal(x_fused, t_fused)


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
#         # debug.print("gamma = {}", gamma)
#         # 将更新后的信号作为下一步输入
#         x = x_new
    
#     return Signal(x, t)


def fdbp1(
    scope: Scope,
    signal,
    steps=3,
    dtaps=261,
    ntaps=41,
    sps=2,
    ixpm_window=3,  # 新增参数，设置IXPM的窗口大小
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
