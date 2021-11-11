# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.ReLU.forward')
def convert__ReLU__forward(ctx: Any, torch_args: Tuple[Any, ...],
                           torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                         ...],
                           trt_kwargs: Dict[str, Any], **kwargs):
    trt_x = get_arg(trt_args, trt_kwargs, 'input', 1)
    layer = ctx.network.add_activation(
        input=trt_x, type=trt.ActivationType.RELU)

    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.nn.functional.relu')
def convert__relu(ctx: Any, torch_args: Tuple[Any, ...],
                  torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                  trt_kwargs: Dict[str, Any], **kwargs):
    trt_x = get_arg(trt_args, trt_kwargs, 'input', 0)
    layer = ctx.network.add_activation(
        input=trt_x, type=trt.ActivationType.RELU)

    return layer.get_output(0)
