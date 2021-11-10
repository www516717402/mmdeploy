from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.functional.max_pool2d')
def convert__max_pool2d(ctx: Any, torch_args: Tuple[Any, ...],
                        torch_kwargs: Dict[str, Any], trt_args: Tuple[Any,
                                                                      ...],
                        trt_kwargs: Dict[str, Any], **kwargs):
    trt_x = get_arg(trt_args, trt_kwargs, 'input', pos=0)
    kernel_size = get_arg(torch_args, torch_kwargs, 'kernel_size', pos=1)
    stride = get_arg(
        torch_args, torch_kwargs, 'stride', pos=2, default=kernel_size)
    padding = get_arg(torch_args, torch_kwargs, 'padding', pos=3, default=0)
    dilation = get_arg(torch_args, torch_kwargs, 'dilation', pos=4, default=1)
    ceil_mode = get_arg(
        torch_args, torch_kwargs, 'ceil_mode', pos=5, default=False)

    # tensorrt require kernel, stride ,padding, dilation must be tuple
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = kernel_size if stride is None else stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    layer = ctx.network.add_pooling(
        input=trt_x, type=trt.PoolingType.MAX, window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.nn.MaxPool2d.forward')
def convert__MaxPool2d__forward(ctx: Any, torch_args: Tuple[Any, ...],
                                torch_kwargs: Dict[str,
                                                   Any], trt_args: Tuple[Any,
                                                                         ...],
                                trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]
    new_torch_args = [
        torch_args[1], module.kernel_size, module.stride, module.padding,
        module.dilation, module.ceil_mode
    ]
    new_trt_args = [
        trt_args[1],
    ]

    return convert__max_pool2d(ctx, new_torch_args, dict(), new_trt_args,
                               dict())
