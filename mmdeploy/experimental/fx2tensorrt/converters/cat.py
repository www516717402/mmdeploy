# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.cat')
def convert__cat(ctx: Any, torch_args: Tuple[Any, ...],
                 torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                 trt_kwargs: Dict[str, Any], **kwargs):
    trt_inputs = get_arg(trt_args, trt_kwargs, 'tensors', pos=0)
    dim = get_arg(torch_args, torch_kwargs, 'dim', pos=1, default=0)
    if dim < 0:
        dim = len(trt_inputs[0].shape) + dim

    layer = ctx.network.add_concatenation(inputs=trt_inputs)

    layer.axis = dim

    return layer.get_output(0)
