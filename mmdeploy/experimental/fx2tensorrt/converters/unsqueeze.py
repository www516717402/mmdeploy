# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, get_trt_shape, new_trt_const


def _convert_unsqueeze_impl(network, trt_input, dim):
    if dim < 0:
        dim = len(trt_input.shape) + dim + 1
    trt_shape = get_trt_shape(network, trt_input)
    trt_one = new_trt_const(network, 1)

    trt_new_shape = [None, trt_one, None]

    if dim == 0:
        trt_new_shape[2] = trt_shape
    elif dim == len(trt_input.shape):
        trt_new_shape[0] = trt_shape
    else:
        trt_new_shape[0] = network.add_slice(trt_shape, [0], [dim],
                                             [1]).get_output(0)
        trt_new_shape[2] = network.add_slice(trt_shape, [dim],
                                             [len(trt_input.shape) - dim],
                                             [1]).get_output(0)

    trt_new_shape = [trt_s for trt_s in trt_new_shape if trt_s is not None]

    trt_new_shape = network.add_concatenation(trt_new_shape).get_output(0)

    layer = network.add_shuffle(trt_input)
    layer.set_input(1, trt_new_shape)
    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.Tensor.unsqueeze')
@TRT_REGISTRY.register_converter('torch.unsqueeze')
def convert__unsqueeze(ctx: Any, torch_args: Tuple[Any, ...],
                       torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                       trt_kwargs: Dict[str, Any], **kwargs):

    trt_input = get_arg(trt_args, trt_kwargs, 'input', pos=0)
    dim = get_arg(torch_args, torch_kwargs, 'dim', pos=1, default=None)

    return _convert_unsqueeze_impl(ctx.network, trt_input, dim)
