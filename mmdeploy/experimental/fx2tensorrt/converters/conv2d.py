from typing import Any, Dict, Tuple

import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, torch_dtype_to_trt


@TRT_REGISTRY.register_converter('torch.nn.Conv2d.forward')
def convert__Conv2d__forward(ctx: Any, torch_args: Tuple[Any, ...],
                             torch_kwargs: Dict[str,
                                                Any], trt_args: Tuple[Any,
                                                                      ...],
                             trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]

    assert isinstance(module,
                      torch.nn.Conv2d), f'layer {module} is not Conv2d.'
    trt_x = get_arg(trt_args, trt_kwargs, 'x', 1)

    # tensorrt require kernel, stride ,padding, dilation must be tuple
    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_convolution(
        input=trt_x,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if module.groups is not None:
        layer.num_groups = module.groups

    return layer.get_output(0)
