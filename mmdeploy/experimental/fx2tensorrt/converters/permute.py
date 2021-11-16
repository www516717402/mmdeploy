# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.Tensor.permute')
def convert__permute(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    """Convert permute to shuffle layer."""
    trt_input = trt_args[0]
    permutation = get_arg(trt_args, trt_kwargs, 'dims', pos=1, default=[])

    # permutation -1 because TRT does not include batch dim
    if isinstance(permutation, int):
        permutation = tuple(torch_args[1:])  # handle permute(a, b, c)
    else:
        permutation = tuple(permutation)  # handle permute([a, b, c])

    layer = ctx.network.add_shuffle(trt_input)
    layer.second_transpose = tuple(permutation)

    return layer.get_output(0)
