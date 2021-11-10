from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_trt_shape, slice_trt_shape


@TRT_REGISTRY.register_converter('torch.Tensor.size')
def convert__Tensor_size(ctx: Any, torch_args: Tuple[Any, ...],
                         torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                       ...],
                         trt_kwargs: Dict[str, Any], **kwargs):
    trt_input = trt_args[0]

    trt_shape = get_trt_shape(ctx.network, trt_input)

    if len(torch_args) < 2:
        return trt_shape

    index = torch_args[1]

    if index < 0:
        index = len(trt_input.shape) + index

    return slice_trt_shape(ctx.network, trt_shape, index, 1)
