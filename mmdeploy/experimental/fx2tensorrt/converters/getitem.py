import torch
from typing import Any, Dict, Tuple

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import slice_trt_shape


def _convert__getitem__torchsize(ctx: Any, torch_args: Tuple[Any, ...],
                                 torch_kwargs: Dict[str, Any],
                                 trt_args: Tuple[Any,
                                                 ...], trt_kwargs: Dict[str,
                                                                        Any]):
    trt_x = trt_args[0]
    index = torch_args[1]
    # process Size
    if isinstance(index, int):
        return slice_trt_shape(ctx.network, trt_x, torch_args[1], 1)
    elif isinstance(index, slice):
        shape_slice = index
        start = 0 if shape_slice.start is None else shape_slice.start
        stop = len(
            torch_args[0]) if shape_slice.stop is None else shape_slice.stop
        size = stop - start
        step = 1 if shape_slice.step is None else shape_slice.step

        return slice_trt_shape(
            ctx.network, trt_x, start=start, size=size, stride=step)
    else:
        raise ValueError(f'Unknown getitem type {type(index)}')


@TRT_REGISTRY.register_converter('operator.getitem')
def convert__getitem(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    x = torch_args[0]

    if isinstance(x, torch.Size):
        return _convert__getitem__torchsize(ctx, torch_args, torch_kwargs,
                                            trt_args, trt_kwargs)
    else:
        raise TypeError(f'Unknown getitem of {type(x)}.')
