# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import concate_trt, get_arg, get_or_new_const


@TRT_REGISTRY.register_converter('torch.Tensor.reshape')
@TRT_REGISTRY.register_converter('torch.Tensor.view')
def convert__view(ctx: Any, torch_args: Tuple[Any, ...],
                  torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                  trt_kwargs: Dict[str, Any], **kwargs):

    trt_input = trt_args[0]
    size = get_arg(trt_args, trt_kwargs, 'shape', pos=1, default=[])

    if isinstance(size, int):
        size = tuple(trt_args[1:])
    elif isinstance(size, trt.ITensor):
        if size.shape[0] == 1:
            size = tuple(trt_args[1:])

    need_dynamic_reshape = isinstance(size, trt.ITensor)

    if any((isinstance(s, trt.ITensor) or s < 0) for s in size):
        need_dynamic_reshape = True
        size = [get_or_new_const(ctx.network, s) for s in size]
        size = concate_trt(ctx.network, *size)

    layer = ctx.network.add_shuffle(trt_input)
    if need_dynamic_reshape:
        layer.set_input(1, size)
    else:
        layer.reshape_dims = tuple(size)

    return layer.get_output(0)
