# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.functional.sigmoid')
@TRT_REGISTRY.register_converter('torch.sigmoid')
@TRT_REGISTRY.register_converter('torch.Tensor.sigmoid')
def convert__sigmoid(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    """Convert sigmoid to activation layer."""
    trt_input = get_arg(trt_args, trt_kwargs, 'input', pos=0)

    layer = ctx.network.add_activation(trt_input, trt.ActivationType.SIGMOID)
    return layer.get_output(0)
