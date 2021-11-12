from typing import Any, Dict, Tuple

import tensorrt as trt
import torch
from torch.fx.node import Argument, map_aggregate

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import new_trt_const


def _map_new_constant(a: Argument, network: trt.INetworkDefinition):
    return map_aggregate(
        a, lambda x: new_trt_const(network, x)
        if isinstance(x, torch.Tensor) else x)


def default_converter(ctx: Any, torch_args: Tuple[Any, ...],
                      torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                      trt_kwargs: Dict[str, Any], torch_output):
    return _map_new_constant(torch_output, ctx.network)


TRT_REGISTRY.set_default_converter(default_converter)
