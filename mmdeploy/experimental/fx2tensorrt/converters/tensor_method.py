from typing import Any, Dict, Tuple

import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import cast_trt_type


@TRT_REGISTRY.register_converter('torch.Tensor.to')
def convert__tensor__to(ctx: Any, torch_args: Tuple[Any, ...],
                        torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                      ...],
                        trt_kwargs: Dict[str, Any], **kwargs):
    torch_to = torch_args[1]

    trt_x = trt_args[0]

    if isinstance(torch_to, torch.device):
        return trt_x
    elif isinstance(torch_to, torch.dtype):
        return cast_trt_type(ctx.network, trt_x, torch_to)
    else:
        raise ValueError(f'Unknown to type: {type(torch_to)}')
