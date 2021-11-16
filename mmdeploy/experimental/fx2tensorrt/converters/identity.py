# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY


@TRT_REGISTRY.register_converter('torch.Tensor.cuda')
@TRT_REGISTRY.register_converter('torch.Tensor.detach')
@TRT_REGISTRY.register_converter('torch.Tensor.contiguous')
@TRT_REGISTRY.register_converter('torch.nn.functional.dropout')
@TRT_REGISTRY.register_converter('torch.nn.functional.dropout2d')
@TRT_REGISTRY.register_converter('torch.nn.functional.dropout3d')
@TRT_REGISTRY.register_converter('torch.Tensor.detach')
def convert__identity(ctx: Any, torch_args: Tuple[Any, ...],
                      torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                      trt_kwargs: Dict[str, Any], **kwargs):
    """support the functions which does not need any conversion."""
    trt_x = trt_args[0]

    return trt_x
