# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, get_trt_shape, new_trt_const


def __convert_squeeze_impl(network, torch_shape, trt_input, dim):
    if dim is None:
        dim = list(
            filter(lambda x: torch_shape[x] == 1, range(len(torch_shape))))
    else:
        if torch_shape[dim] != 1:
            return trt_input
        if dim < 0:
            dim = len(torch_shape) + dim
        dim = [dim]

    reverse_dim = list(filter(lambda x: x not in dim, range(len(torch_shape))))
    trt_reverse_dim = new_trt_const(
        network, torch.tensor(reverse_dim, dtype=torch.int32))

    trt_shape = get_trt_shape(network, trt_input)
    trt_new_shape = network.add_gather(trt_shape, trt_reverse_dim,
                                       0).get_output(0)

    layer = network.add_shuffle(trt_input)
    layer.set_input(1, trt_new_shape)
    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.Tensor.squeeze')
@TRT_REGISTRY.register_converter('torch.squeeze')
def convert__squeeze(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    torch_input = get_arg(torch_args, torch_kwargs, 'input', pos=0)
    trt_input = get_arg(trt_args, trt_kwargs, 'input', pos=0)
    dim = get_arg(torch_args, torch_kwargs, 'dim', pos=1, default=None)
    torch_shape = torch_input.shape
    return __convert_squeeze_impl(ctx.network, torch_shape, trt_input, dim)
