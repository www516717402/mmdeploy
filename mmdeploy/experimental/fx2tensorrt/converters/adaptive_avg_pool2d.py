# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
import logging
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.functional.adaptive_avg_pool2d')
def convert__adaptive_avg_pool2d(ctx: Any, torch_args: Tuple[Any, ...],
                                 torch_kwargs: Dict[str,
                                                    Any], trt_args: Tuple[Any,
                                                                          ...],
                                 trt_kwargs: Dict[str, Any], **kwargs):
    x = get_arg(torch_args, torch_kwargs, 'input', 0)
    output_size = get_arg(torch_args, torch_kwargs, 'output_size', 1)
    trt_x = get_arg(trt_args, trt_kwargs, 'input', 0)
    trt_output_size = get_arg(trt_args, trt_kwargs, 'output_size', 1)

    if isinstance(trt_output_size, trt.ITensor):
        logging.warning('Dynamic output_size of adaptive_avg_pool2d '
                        'might cause unexpected behavior.')

    # tensorrt require kernel, stride ,padding, dilation must be tuple
    if not isinstance(output_size, tuple):
        output_size = (output_size, ) * 2

    if output_size[0] == 1 and output_size[1] == 1:
        # global pooling
        shape_length = len(trt_x.shape)
        axes = (1 << (shape_length - 1)) + (1 << (shape_length - 2))
        keepdim = True
        layer = ctx.network.add_reduce(trt_x, trt.ReduceOperation.AVG, axes,
                                       keepdim)
    else:
        if trt_x.shape[-2] < 0 or trt_x.shape[-1] < 0:
            logging.warning('dynamic adaptive pool might cause '
                            'unexpected behavior')
        stride = (x.shape[-2] // output_size[-2],
                  x.shape[-1] // output_size[-1])
        kernel_size = [
            x.shape[-2] - (output_size[-2] - 1) * stride[0],
            x.shape[-1] - (output_size[-1] - 1) * stride[1]
        ]
        layer = ctx.network.add_pooling(
            input=trt_x, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
        layer.stride = stride

    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert__AdaptiveAvgPool2d__forward(ctx: Any, torch_args: Tuple[Any, ...],
                                        torch_kwargs: Dict[str, Any],
                                        trt_args: Tuple[Any, ...],
                                        trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]

    new_torch_args = [torch_args[1], module.output_size]
    new_trt_args = [trt_args[1], module.output_size]

    return convert__adaptive_avg_pool2d(ctx, new_torch_args, dict(),
                                        new_trt_args, dict())
