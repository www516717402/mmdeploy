# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg, torch_dtype_to_trt


@TRT_REGISTRY.register_converter('torch.nn.functional.linear')
def convert__linear(ctx: Any, torch_args: Tuple[Any, ...],
                    torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                    trt_kwargs: Dict[str, Any], **kwargs):
    trt_x = get_arg(trt_args, trt_kwargs, 'input', 0)
    weight = get_arg(torch_args, torch_kwargs, 'weight', pos=1)
    bias = get_arg(torch_args, torch_kwargs, 'bias', pos=2, default=None)

    origin_shape = trt_x.shape

    # reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(trt_x)
    layer.reshape_dims = (0, ) * len(origin_shape) + (1, 1)
    trt_x = layer.get_output(0)

    # add fully connected
    if bias is not None:
        bias = bias.detach().cpu().numpy()
    else:
        bias = trt.Weights(torch_dtype_to_trt(weight.dtype))

    weight = weight.detach().cpu().numpy()

    layer = ctx.network.add_convolution(
        input=trt_x,
        num_output_maps=weight.shape[0],
        kernel_shape=(1, 1),
        kernel=weight,
        bias=bias)

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    # layer.reshape_dims = tuple(output.shape[1:])
    layer.reshape_dims = (0, ) * len(origin_shape)

    return layer.get_output(0)


@TRT_REGISTRY.register_converter('torch.nn.Linear.forward')
def convert__Linear__forward(ctx: Any, torch_args: Tuple[Any, ...],
                             torch_kwargs: Dict[str,
                                                Any], trt_args: Tuple[Any,
                                                                      ...],
                             trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]

    new_torch_args = [torch_args[1], module.weight, module.bias]
    new_trt_args = [trt_args[1], module.weight, module.bias]

    return convert__linear(ctx, new_torch_args, dict(), new_trt_args, dict())
