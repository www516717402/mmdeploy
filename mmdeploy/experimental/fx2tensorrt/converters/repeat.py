# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import (get_or_new_const, get_trt_shape, new_trt_const,
                               slice_trt_shape)


def __unsqueeze_input(network, trt_input, dim):
    if dim == len(trt_input.shape):
        return trt_input
    ones_trt = new_trt_const(
        network, torch.ones(dim - len(trt_input.shape), dtype=torch.int32))
    trt_input_shape = get_trt_shape(network, trt_input)
    trt_input_shape = network.add_concatenation([ones_trt, trt_input_shape
                                                 ]).get_output(0)
    layer = network.add_shuffle(trt_input)
    layer.set_input(1, trt_input_shape)
    trt_input = layer.get_output(0)
    return trt_input


def __convert_repeat_impl(network, trt_input, trt_output_shape):
    dim = trt_output_shape.shape[0]

    if len(trt_input.shape) < dim:
        trt_input = __unsqueeze_input(network, trt_input, dim)

    zeros_trt = new_trt_const(network, torch.zeros(dim, dtype=torch.int32))
    ones_trt = new_trt_const(network, torch.ones(dim, dtype=torch.int32))

    layer = network.add_slice(trt_input, [0] * dim, [1] * dim, [1] * dim)
    layer.set_input(1, zeros_trt)
    layer.set_input(2, trt_output_shape)
    layer.set_input(3, ones_trt)
    layer.mode = trt.SliceMode.WRAP

    output_trt = layer.get_output(0)

    return output_trt


@TRT_REGISTRY.register_converter('torch.Tensor.repeat')
def convert__repeat(ctx: Any, torch_args: Tuple[Any, ...],
                    torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                    trt_kwargs: Dict[str, Any], **kwargs):
    trt_input = trt_args[0]
    repeats = trt_args[1]
    if isinstance(repeats, (int, trt.ITensor)):
        repeats = trt_args[1:]

    trt_input = __unsqueeze_input(ctx, trt_input, len(repeats))

    # compute output shape
    trt_input_shape = get_trt_shape(ctx.network, trt_input)
    trt_repeat_times = [get_or_new_const(ctx.network, rep) for rep in repeats]
    trt_repeat_times = ctx.network.add_concatenation(
        trt_repeat_times).get_output(0)

    trt_output_shape = ctx.network.add_elementwise(
        trt_input_shape, trt_repeat_times,
        trt.ElementWiseOperation.PROD).get_output(0)

    # convert repeat
    return __convert_repeat_impl(ctx.network, trt_input, trt_output_shape)


@TRT_REGISTRY.register_converter('torch.Tensor.expand')
def convert__expand(ctx: Any, torch_args: Tuple[Any, ...],
                    torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                    trt_kwargs: Dict[str, Any], **kwargs):
    torch_input = torch_args[0]

    trt_input = trt_args[0]
    sizes = trt_args[1]
    if isinstance(sizes, (int, trt.ITensor)):
        sizes = trt_args[1:]
    dim = len(sizes)

    # unsqueeze if necessary
    if torch_input.dim() < dim:
        trt_input = __unsqueeze_input(ctx.network, trt_input, dim)
    trt_input_shape = get_trt_shape(ctx.network, trt_input)

    # compute output shape
    trt_output_shape = []
    for i, s in enumerate(sizes):
        if isinstance(s, trt.ITensor):
            trt_s = s
        else:
            trt_s = new_trt_const(ctx.network, s)

        if isinstance(s, trt.ITensor) or (isinstance(s, int) and s > 0):
            trt_output_shape.append(trt_s)
        else:
            trt_output_shape.append(
                slice_trt_shape(ctx.network, trt_input_shape, i, 1))

    trt_output_shape = ctx.network.add_concatenation(
        trt_output_shape).get_output(0)

    # convert repeat
    return __convert_repeat_impl(ctx.network, trt_input, trt_output_shape)
