# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, get_trt_shape, slice_trt_shape


@TRT_REGISTRY.register_converter('torch.flatten')
@TRT_REGISTRY.register_converter('torch.Tensor.flatten')
def convert__flatten(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    trt_input = get_arg(trt_args, trt_kwargs, 'input', 0)
    start_dim = get_arg(
        torch_args, torch_kwargs, 'start_dim', pos=1, default=0)
    end_dim = get_arg(torch_args, torch_kwargs, 'end_dim', pos=2, default=-1)

    if start_dim == -1:
        start_dim = len(trt_input.shape) - 1
    if end_dim == -1:
        end_dim = len(trt_input.shape) - 1

    if start_dim == end_dim:
        # no need to flatten
        return trt_input

    trt_shape = get_trt_shape(ctx.network, trt_input)

    trt_new_shape = [None] * 3

    if start_dim != 0:
        trt_new_shape[0] = slice_trt_shape(ctx.network, trt_shape, 0,
                                           start_dim)
    if end_dim != len(trt_input.shape) - 1:
        trt_new_shape[2] = slice_trt_shape(ctx.network, trt_shape, end_dim + 1,
                                           len(trt_input.shape) - end_dim - 1)

    trt_new_shape[1] = slice_trt_shape(ctx.network, trt_shape, start_dim,
                                       end_dim - start_dim + 1)
    trt_mid = slice_trt_shape(ctx.network, trt_new_shape[1], 0, 1)
    for i in range(end_dim - start_dim):
        trt_other = slice_trt_shape(ctx.network, trt_new_shape[1], i + 1, 1)
        trt_mid = ctx.network.add_elementwise(
            trt_mid, trt_other, trt.ElementWiseOperation.PROD).get_output(0)

    trt_new_shape[1] = trt_mid

    trt_new_shape = [x for x in trt_new_shape if x is not None]

    trt_new_shape = ctx.network.add_concatenation(trt_new_shape).get_output(0)

    layer = ctx.network.add_shuffle(trt_input)
    layer.set_input(1, trt_new_shape)

    return layer.get_output(0)
