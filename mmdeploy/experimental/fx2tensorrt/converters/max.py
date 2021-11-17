# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, torch_dim_to_trt_axes
from .topk import __convert_topk_impl
from .squeeze import __convert_squeeze_impl


def __convert_max_elementwise(network, trt_args, trt_kwargs):
    trt_a = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'other', 1)
    layer = network.add_elementwise(trt_a, trt_b, trt.ElementWiseOperation.MAX)
    return layer.get_output(0)


def __convert_max_reduce(network, torch_args, torch_kwargs, trt_args,
                         trt_kwargs):
    torch_input = get_arg(torch_args, torch_kwargs, 'input', 0)
    dim = get_arg(torch_args, torch_kwargs, 'dim', 1)
    keepdim = get_arg(torch_args, torch_kwargs, 'keepdim', 2)
    trt_input = get_arg(trt_args, trt_kwargs, 'input', 0)

    if dim is None:
        dim = tuple(range(0, torch_input.ndim))
        layer = network.add_reduce(trt_input, trt.ReduceOperation.MAX,
                                   torch_dim_to_trt_axes(dim), keepdim)
        return layer.get_output(0)
    else:
        trt_value, trt_index = __convert_topk_impl(network, trt_input, 1, dim)

        if not keepdim and len(trt_index.shape) > 1:
            torch_shape = list(torch_input.shape)
            torch_shape[dim] = 1
            trt_value = __convert_squeeze_impl(network, torch_shape, trt_value,
                                               dim)
            trt_index = __convert_squeeze_impl(network, torch_shape, trt_index,
                                               dim)

        return trt_value, trt_index


@TRT_REGISTRY.register_converter('torch.max')
@TRT_REGISTRY.register_converter('torch.Tensor.max')
def convert__max(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    """convert torch max with elementwise or reduce"""
    if len(torch_args) > 1 and isinstance(torch_args[1], torch.Tensor):
        return __convert_max_elementwise(ctx.network, trt_args, trt_kwargs)
    else:
        return __convert_max_reduce(ctx.network, torch_args, torch_kwargs,
                                    trt_args, trt_kwargs)
