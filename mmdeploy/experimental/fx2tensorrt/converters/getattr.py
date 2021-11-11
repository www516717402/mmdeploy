# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_trt_shape


@TRT_REGISTRY.register_converter('getattr')
def convert__getattr(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    x = torch_args[0]
    attr_name = torch_args[1]

    if isinstance(x, torch.Tensor) and attr_name == 'shape':
        trt_x = trt_args[0]
        trt_shape = get_trt_shape(ctx.network, trt_x)
        return trt_shape
    else:
        raise ValueError(f'Unsupported attrbute name:{attr_name}')
