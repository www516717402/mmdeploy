# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.functional.avg_pool2d')
def convert__avg_pool2d(ctx: Any, torch_args: Tuple[Any, ...],
                        torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                      ...],
                        trt_kwargs: Dict[str, Any], **kwargs):
    trt_x = get_arg(trt_args, trt_kwargs, 'input', 0)
    kernel_size = get_arg(torch_args, torch_kwargs, 'kernel_size', 1)
    stride = get_arg(torch_args, torch_kwargs, 'stride', 2, kernel_size)
    padding = get_arg(torch_args, torch_kwargs, 'padding', 3, 0)
    ceil_mode = get_arg(
        torch_args, torch_kwargs, 'ceil_mode', pos=4, default=False)
    count_include_pad = get_arg(
        torch_args, torch_kwargs, 'count_include_pad', pos=5, default=True)

    # tensorrt require kernel, stride ,padding, dilation must be tuple
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = kernel_size if stride is None else stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=trt_x, type=trt.PoolingType.AVERAGE, window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding

    layer.average_count_excludes_padding = not count_include_pad
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.nn.AvgPool2d.forward')
def convert__AvgPool2d__forward(ctx: Any, torch_args: Tuple[Any, ...],
                                torch_kwargs: Dict[str,
                                                   Any], trt_args: Tuple[Any,
                                                                         ...],
                                trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]

    new_torch_args = [
        torch_args[1], module.kernel_size, module.stride, module.padding
    ]
    new_trt_args = [
        trt_args[1],
    ]

    return convert__avg_pool2d(ctx, new_torch_args, dict(), new_trt_args,
                               dict())
