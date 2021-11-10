import logging
from typing import Any, Dict, Tuple

import tensorrt as trt
from packaging import version

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import concate_trt, get_arg, get_trt_shape


@TRT_REGISTRY.register_converter('torch.nn.functional.interpolate')
def convert__interpolate(ctx: Any, torch_args: Tuple[Any, ...],
                         torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                       ...],
                         trt_kwargs: Dict[str, Any], torch_output: Any):

    trt_input = get_arg(trt_args, trt_kwargs, 'input', pos=0)
    trt_size = get_arg(trt_args, trt_kwargs, 'size', pos=1, default=None)

    size = get_arg(torch_args, torch_kwargs, 'size', pos=1, default=None)
    scale_factor = get_arg(
        torch_args, torch_kwargs, 'scale_factor', pos=2, default=None)
    mode = get_arg(torch_args, torch_kwargs, 'mode', pos=3, default='nearest')
    align_corners = get_arg(
        torch_args, torch_kwargs, 'align_corners', pos=4, default=None)

    if isinstance(scale_factor, int):
        scale_factor = float(scale_factor)
    if isinstance(scale_factor, float):
        scale_factor = tuple([scale_factor] * (len(input.shape) - 2))

    if isinstance(size, int):
        size = [size]

    if align_corners is None:
        align_corners = False

    trt_new_shape = None
    if isinstance(trt_size, trt.ITensor):
        trt_pre_input_size = get_trt_shape(ctx.network, trt_input, 0,
                                           (len(trt_input.shape) - len(size)))
        trt_new_shape = concate_trt(ctx.network, trt_pre_input_size, trt_size)

    layer = ctx.network.add_resize(trt_input)

    if trt_new_shape is not None:
        # dynamic size
        layer.set_input(1, trt_new_shape)
    elif scale_factor is not None:
        scale_factor = (1, ) * 2 + tuple(scale_factor)
        layer.scales = scale_factor
    else:
        layer.shape = tuple(torch_output.shape)

    if version.parse(trt.__version__) >= version.parse('8'):
        layer.coordinate_transformation = \
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        layer.nearest_rounding = trt.ResizeRoundMode.HALF_DOWN
    else:
        layer.align_corners = align_corners

    if mode == 'nearest':
        layer.resize_mode = trt.ResizeMode.NEAREST
    elif mode == 'linear':
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.LINEAR
        logging.warning(
            f'Unsupported resize_mode: {mode}, use linear instead.')

    return layer.get_output(0)
