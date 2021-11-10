from typing import Any, Dict, Tuple

import numpy as np
import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_arg


@TRT_REGISTRY.register_converter('torch.nn.BatchNorm2d.forward')
def convert__BatchNorm2d__forward(ctx: Any, torch_args: Tuple[Any, ...],
                                  torch_kwargs: Dict[str, Any],
                                  trt_args: Tuple[Any, ...],
                                  trt_kwargs: Dict[str, Any], **kwargs):
    module = torch_args[0]

    assert isinstance(
        module, torch.nn.BatchNorm2d), f'layer {module} is not BatchNorm2d.'
    trt_x = get_arg(trt_args, trt_kwargs, 'x', 1)

    scale = module.weight.detach().cpu().numpy() / np.sqrt(
        module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy(
    ) - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    layer = ctx.network.add_scale(trt_x, trt.ScaleMode.CHANNEL, bias, scale,
                                  power)

    return layer.get_output(0)
