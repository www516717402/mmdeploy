# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import (align_trt_dims, cast_trt_type, get_arg,
                               new_trt_const_like)


def _convert_elementwise_impl(network, torch_a, torch_b, trt_a, trt_b,
                              elementwise_op):
    assert isinstance(trt_a, trt.ITensor) or isinstance(trt_b, trt.ITensor),\
        'One of input of binary ops should be tensor.'\
        + f' get type: {type(trt_a)}, {type(trt_b)}'

    if not isinstance(trt_a, trt.ITensor):
        trt_a = new_trt_const_like(network, torch_a, torch_b)

    if not isinstance(trt_b, trt.ITensor):
        trt_b = new_trt_const_like(network, torch_b, torch_a)

    # one of them might be shape input
    if trt_a.dtype != trt_b.dtype:
        if not isinstance(torch_a, torch.Tensor):
            trt_a = cast_trt_type(network, trt_a, trt_b.dtype)
        if not isinstance(torch_b, torch.Tensor):
            trt_b = cast_trt_type(network, trt_b, trt_a.dtype)

    trt_a, trt_b = align_trt_dims(network, trt_a, trt_b)

    trt_output = network.add_elementwise(trt_a, trt_b,
                                         elementwise_op).get_output(0)
    return trt_output


@TRT_REGISTRY.register_converter('torch.Tensor.add')
@TRT_REGISTRY.register_converter('torch.add')
def convert__add(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'input', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'other', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'other', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.SUM)


@TRT_REGISTRY.register_converter('operator.add')
def convert__operator_add(ctx: Any, torch_args: Tuple[Any, ...],
                          torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                        ...],
                          trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'a', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'b', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'a', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'b', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.SUM)


@TRT_REGISTRY.register_converter('torch.Tensor.sub')
@TRT_REGISTRY.register_converter('torch.sub')
def convert__sub(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'input', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'other', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'other', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.SUB)


@TRT_REGISTRY.register_converter('operator.sub')
def convert__operator_sub(ctx: Any, torch_args: Tuple[Any, ...],
                          torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                        ...],
                          trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'a', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'b', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'a', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'b', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.SUB)


@TRT_REGISTRY.register_converter('torch.Tensor.mul')
@TRT_REGISTRY.register_converter('torch.mul')
def convert__mul(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'input', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'other', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'other', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.PROD)


@TRT_REGISTRY.register_converter('operator.mul')
def convert__operator_mul(ctx: Any, torch_args: Tuple[Any, ...],
                          torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                        ...],
                          trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'a', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'b', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'a', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'b', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.PROD)


@TRT_REGISTRY.register_converter('torch.Tensor.div')
@TRT_REGISTRY.register_converter('torch.div')
def convert__div(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'input', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'other', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'other', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.DIV)


@TRT_REGISTRY.register_converter('operator.truediv')
def convert__operator_truediv(ctx: Any, torch_args: Tuple[Any, ...],
                              torch_kwargs: Dict[str,
                                                 Any], trt_args: Tuple[Any,
                                                                       ...],
                              trt_kwargs: Dict[str, Any], **kwargs):
    torch_a = get_arg(torch_args, torch_kwargs, 'a', 0)
    torch_b = get_arg(torch_args, torch_kwargs, 'b', 1)
    trt_a = get_arg(trt_args, trt_kwargs, 'a', 0)
    trt_b = get_arg(trt_args, trt_kwargs, 'b', 1)
    return _convert_elementwise_impl(ctx.network, torch_a, torch_b, trt_a,
                                     trt_b, trt.ElementWiseOperation.DIV)
