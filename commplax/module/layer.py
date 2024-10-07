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


from flax.core import Scope, init, apply
from functools import reduce
from commplax.util import wrapped_partial as partial
from typing import NamedTuple, Optional, Any
from commplax.module import core
import operator


class Layer(NamedTuple):
    name: Optional[str]
    init: Any
    apply: Any
    core: Any
    mutable: Any


# def make_layer(f, mutable=()):
#     def _layer(layer_transform=lambda f: f, **kwargs):
#         name = kwargs.pop('name', None)
#         core_fun = layer_transform(partial(f, **kwargs))

#         def init_fun(rng, *args, **kwargs):
#             return init(core_fun)(rng, *args, **kwargs)

#         def apply_fun(params, *args, **kwargs):
#             return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

#         return Layer(name, init_fun, apply_fun, core_fun, mutable)
#     return _layer

def make_layer(f, mutable=()):
    def _layer(*args, **kwargs):
        name = kwargs.pop('name', None)
        layer_transform = kwargs.pop('layer_transform', lambda f: f)
        core_fun = layer_transform(partial(f, *args, **kwargs))

        def init_fun(rng, *init_args, **init_kwargs):
            return init(core_fun)(rng, *init_args, **init_kwargs)

        def apply_fun(params, *apply_args, **apply_kwargs):
            return apply(core_fun, mutable=mutable)(params, *apply_args, **apply_kwargs)

        return Layer(name, init_fun, apply_fun, core_fun, mutable)
    return _layer



def vmap(layer, **vmap_kwargs):
    return partial(layer, layer_transform=partial(core.vmap, **vmap_kwargs))


def scan(layer, **scan_kwargs):
    return partial(layer, layer_transform=partial(core.scan, **scan_kwargs))


BatchPowerNorm = make_layer(core.batchpowernorm, mutable=('norm',))
Conv1d = make_layer(core.conv1d)
MIMOConv1d = make_layer(core.mimoconv1d)
MIMOAF = make_layer(core.mimoaf, mutable=('af_state',))
FDBP = make_layer(core.fdbp)

FDBP1 = make_layer(core.fdbp1)
BatchPowerNorm1 = make_layer(core.batchpowernorm1, mutable=('norm',))
Conv1d1 = make_layer(core.conv1d1)
MIMOAF1 = make_layer(core.mimoaf1, mutable=('af_state',))
MIMOFOEAf1 = make_layer(core.mimofoeaf1, mutable=('af_state',))

SimpleFn = make_layer(core.simplefn)
MIMOFOEAf = make_layer(core.mimofoeaf, mutable=('af_state',))
Identity = make_layer(core.identity)
FanOut = make_layer(core.fanout)
FanInSum = make_layer(core.fanin_sum)
FanInMean = make_layer(core.fanin_mean)
# def Serial(*layers, name='serial'):
#     names, _, _, core_funs, mutables = zip(*layers)
#     core_fun = core.serial(*zip(names, core_funs))
#     mutable = reduce(operator.add, list(mutables))

#     def init_fun(rng, *args, **kwargs):
#         return init(core_fun)(rng, *args, **kwargs)

#     def apply_fun(params, *args, **kwargs):
#         return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

#     return Layer(name, init_fun, apply_fun, core_fun, mutable)


# def Parallel(*layers, name='parallel'):
#     names, _, _, core_funs, mutables = zip(*layers)
#     core_fun = core.serial(*zip(names, core_funs))
#     mutable = reduce(operator.add, list(mutables))

#     def init_fun(rng, *args, **kwargs):
#         return init(core_fun)(rng, *args, **kwargs)

#     def apply_fun(params, *args, **kwargs):
#         return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

#     return Layer(name, init_fun, apply_fun, core_fun, mutable)
def Serial(*layers, name='serial'):
    names, _, _, core_funs, mutables = zip(*layers)
    core_fun = core.serial(*zip(names, core_funs))
    mutable = reduce(operator.add, list(mutables))

    def init_fun(rng, *args, **kwargs):
        return init(core_fun)(rng, *args, **kwargs)

    def apply_fun(params, *args, **kwargs):
        return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

    return Layer(name, init_fun, apply_fun, core_fun, mutable)

def Parallel(*layers, name='parallel'):
    layer_objs = []
    for item in layers:
        if isinstance(item, Layer):
            # 如果是 Layer 对象，直接使用
            layer_obj = item
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Layer):
            # 如果是 (name, Layer) 形式
            layer_name, layer_obj = item
            # 覆盖 Layer 对象的 name 属性
            layer_obj = layer_obj._replace(name=layer_name)
        else:
            raise ValueError(f"Invalid layer format in Parallel: {item}")
        layer_objs.append(layer_obj)

    names, inits, applies, core_funs, mutables = zip(*layer_objs)
    core_fun = core.parallel(*zip(names, core_funs))
    mutable = reduce(operator.add, list(mutables))

    def init_fun(rng, inputs, **kwargs):
        return init(core_fun)(rng, inputs, **kwargs)

    def apply_fun(params, inputs, **kwargs):
        return apply(core_fun, mutable=mutable)(params, inputs, **kwargs)

    return Layer(name, init_fun, apply_fun, core_fun, mutable)




