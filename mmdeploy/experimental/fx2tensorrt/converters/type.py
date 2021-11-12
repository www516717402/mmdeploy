# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import cast_trt_type, get_arg


@TRT_REGISTRY.register_converter('torch.Tensor.type_as')
def convert__tensor__type_as(ctx: Any, torch_args: Tuple[Any, ...],
                             torch_kwargs: Dict[str,
                                                Any], trt_args: Tuple[Any,
                                                                      ...],
                             trt_kwargs: Dict[str, Any], **kwargs):

    trt_input = trt_args[0]
    trt_other = get_arg(trt_args, trt_kwargs, 'tensor', pos=1, default=[])

    return cast_trt_type(ctx.network, trt_input, trt_other.dtype)


@TRT_REGISTRY.register_converter('torch.Tensor.type')
def convert__tensor__type(ctx: Any, torch_args: Tuple[Any, ...],
                          torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                        ...],
                          trt_kwargs: Dict[str, Any], **kwargs):

    trt_input = trt_args[0]
    dtype = get_arg(trt_args, trt_kwargs, 'dtype', pos=1, default=[])

    return cast_trt_type(ctx.network, trt_input, dtype)
