# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
import logging
from typing import Any, Dict, Tuple

import tensorrt as trt

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


def __convert_topk_impl(network, trt_input, k, dim, largest=True):
    if dim < 0:
        dim += len(trt_input.shape)
    topkOp = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN

    # can only use topk on dim>=2
    need_unsqueeze = len(trt_input.shape) == 1

    if need_unsqueeze:
        layer = network.add_shuffle(trt_input)
        layer.reshape_dims = (1, ) + tuple(trt_input.shape)
        trt_input = layer.get_output(0)
        dim += 1

    layer = network.add_topk(trt_input, topkOp, k, 1 << dim)

    trt_output0 = layer.get_output(0)
    trt_output1 = layer.get_output(1)

    # recovery
    if need_unsqueeze:
        layer = network.add_shuffle(trt_output0)
        layer.reshape_dims = tuple(trt_output0.shape)[1:]
        trt_output0 = layer.get_output(0)

        layer = network.add_shuffle(trt_output1)
        layer.reshape_dims = tuple(trt_output1.shape)[1:]
        trt_output1 = layer.get_output(0)

    return trt_output0, trt_output1


@TRT_REGISTRY.register_converter('torch.topk')
@TRT_REGISTRY.register_converter('torch.Tensor.topk')
def convert__topk(ctx: Any, torch_args: Tuple[Any, ...],
                  torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                  trt_kwargs: Dict[str, Any], **kwargs):
    """convert topk to TensorRT topk layer"""
    trt_input = trt_args[0]
    k = get_arg(torch_args, torch_kwargs, 'k', pos=1)
    dim = get_arg(
        torch_args,
        torch_kwargs,
        'dim',
        pos=2,
        default=len(trt_input.shape) - 1)

    if k > 3840:
        logging.warning(
            f'topk = {k} > 3840 is not allowed in TensorRT, use 3840 instead.')
        k = 3840

    if dim is None:
        dim = len(trt_input.shape) - 1
    if dim < 0:
        dim += len(trt_input.shape)

    largest = get_arg(torch_args, torch_kwargs, 'largest', pos=3, default=True)

    return __convert_topk_impl(ctx.network, trt_input, k, dim, largest)
