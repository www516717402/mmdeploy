import logging
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import tensorrt as trt
import torch
from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Node, Target, map_aggregate

from .converter_registry import TRT_REGISTRY
from .converter_utils import new_trt_const, torch_dtype_to_trt


class _TensorPair:
    """A wrap class of tensor and engine meta info."""

    def __init__(self, tensor: torch.Tensor, meta: Any) -> None:
        self.tensor = tensor
        self.meta = meta


def map_tensor_pair(a: Argument, fn: Callable) -> Argument:
    """map _TensorPair with given transform function."""
    return map_aggregate(a, lambda x: fn(x)
                         if isinstance(x, _TensorPair) else x)


def _mark_network_outputs(a: Argument, network: trt.INetworkDefinition,
                          name_iter: Iterator) -> None:
    """mark outputs in network."""

    def mark_output(x):
        x.name = next(name_iter)
        network.mark_output(x)

    map_aggregate(
        a, lambda x: mark_output(x) if isinstance(x, trt.ITensor) else x)


class TRTFXParser(Interpreter):
    """Parser class used to trace the graph and generate tensorrt engine."""

    def __init__(self,
                 module: GraphModule,
                 network: trt.INetworkDefinition,
                 profile: trt.IOptimizationProfile,
                 input_names: Optional[Sequence[str]] = None,
                 output_names: Optional[Sequence[str]] = None):
        super().__init__(module)
        self.network = network
        self.profile = profile
        self.input_names = [] if input_names is None else input_names
        self.output_names = [] if output_names is None else output_names

    def get_network_input_shape(self, target: 'Target',
                                tensor: torch.Tensor) -> list:
        """get the input shape with -1 in dynamic axis."""
        opt_shapes = self.profile.get_shape(str(target))
        tensor_shape = tensor.shape
        ret_shape = list(tensor_shape)
        if opt_shapes is None or len(opt_shapes) <= 0:
            return ret_shape

        for i in range(len(ret_shape)):
            min_s = opt_shapes[0][i]
            opt_s = opt_shapes[1][i]
            max_s = opt_shapes[2][i]

            old_s = tensor_shape[i]

            assert min_s <= old_s and max_s >= old_s, \
                'input tensor does not match profile'

            if min_s != max_s or min_s != opt_s:
                ret_shape[i] = -1
        return ret_shape

    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None) -> Any:
        """start trace and generate network."""
        self.input_name_iter = iter(self.input_names)
        return super().run(*args, initial_env=initial_env)

    def placeholder(self, target: 'Target', args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any]) -> Any:
        logging.debug(f'Create placeholder: {target}')
        torch_input = super().placeholder(target, args, kwargs)

        # process network inputs
        if isinstance(torch_input, torch.Tensor):
            name = next(self.input_name_iter, str(target))
            trt_input = self.network.add_input(
                name=name,
                shape=self.get_network_input_shape(name, torch_input),
                dtype=torch_dtype_to_trt(torch_input.dtype))
            return _TensorPair(torch_input, trt_input)
        else:
            num_inputs = len(torch_input)
            new_input = []
            for i in range(num_inputs):
                new_name = next(self.input_name_iter, f'{str(target)}_{i}')
                trt_input_shape = self.get_network_input_shape(
                    new_name, torch_input[i])
                trt_input_dtype = torch_dtype_to_trt(torch_input[i].dtype)
                trt_input = self.network.add_input(
                    name=new_name,
                    shape=trt_input_shape,
                    dtype=trt_input_dtype)
                new_input.append(_TensorPair(torch_input[i], trt_input))
            return new_input

    def get_attr(self, target: 'Target', args: Tuple[Argument, ...],
                 kwargs: Dict[str, Any]) -> Any:
        """process get_attr node."""
        logging.debug(f'Create get_attr: {target}')
        torch_args = map_tensor_pair(args, lambda arg: arg.tensor)
        torch_kwargs = map_tensor_pair(kwargs, lambda arg: arg.tensor)
        torch_output = super().get_attr(target, torch_args, torch_kwargs)

        # create constant
        trt_output = None
        if isinstance(torch_output, torch.Tensor):
            trt_output = new_trt_const(self.network, torch_output)
        return _TensorPair(torch_output, trt_output)

    def call_function(self, target: 'Target', args: Tuple[Argument, ...],
                      kwargs: Dict[str, Any]) -> Any:
        """process call_function node."""
        logging.debug(f'Create call_function: {target}')

        torch_args = map_tensor_pair(args, lambda arg: arg.tensor)
        torch_kwargs = map_tensor_pair(kwargs, lambda arg: arg.tensor)
        torch_output = super().call_function(target, torch_args, torch_kwargs)
        # run converter
        trt_args = map_tensor_pair(args, lambda arg: arg.meta)
        trt_kwargs = map_tensor_pair(kwargs, lambda arg: arg.meta)
        converter = TRT_REGISTRY.get_converter(target)
        trt_output = converter(
            self,
            torch_args,
            torch_kwargs,
            trt_args,
            trt_kwargs,
            torch_output=torch_output)
        return _TensorPair(torch_output, trt_output)

    def call_method(self, target: 'Target', args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any]) -> Any:
        """process call_method node."""
        logging.debug(f'Create call_method: {target}')
        torch_args = map_tensor_pair(args, lambda arg: arg.tensor)
        torch_kwargs = map_tensor_pair(kwargs, lambda arg: arg.tensor)
        torch_output = super().call_method(target, torch_args, torch_kwargs)

        # run converter
        trt_args = map_tensor_pair(args, lambda arg: arg.meta)
        trt_kwargs = map_tensor_pair(kwargs, lambda arg: arg.meta)
        self_obj, *args_tail = torch_args
        method_func = getattr(type(self_obj), target)
        converter = TRT_REGISTRY.get_converter(method_func)
        trt_output = converter(
            self,
            torch_args,
            torch_kwargs,
            trt_args,
            trt_kwargs,
            torch_output=torch_output)
        return _TensorPair(torch_output, trt_output)

    def call_module(self, target: 'Target', args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any]) -> Any:
        """process call_module node."""
        logging.debug(f'Create call_module: {target}')
        # run torch
        torch_args = map_tensor_pair(args, lambda arg: arg.tensor)
        torch_kwargs = map_tensor_pair(kwargs, lambda arg: arg.tensor)
        torch_output = super().call_module(target, torch_args, torch_kwargs)

        # run converter
        trt_args = map_tensor_pair(args, lambda arg: arg.meta)
        trt_kwargs = map_tensor_pair(kwargs, lambda arg: arg.meta)
        submod = self.fetch_attr(target)
        converter = TRT_REGISTRY.get_converter(submod.forward.__func__)
        torch_args = (submod, ) + torch_args
        trt_args = (submod, ) + trt_args
        trt_output = converter(
            self,
            torch_args,
            torch_kwargs,
            trt_args,
            trt_kwargs,
            torch_output=torch_output)
        return _TensorPair(torch_output, trt_output)

    def output(self, target: 'Target', args: Tuple[Argument, ...],
               kwargs: Dict[str, Any]) -> Any:
        """process output node."""
        logging.debug(f'Create output: {target}')
        torch_args = map_tensor_pair(args, lambda arg: arg.tensor)
        torch_kwargs = map_tensor_pair(kwargs, lambda arg: arg.tensor)
        torch_output = super().output(target, torch_args, torch_kwargs)

        trt_args = map_tensor_pair(args, lambda arg: arg.meta)

        output_name_iter = iter(self.output_names)
        _mark_network_outputs(trt_args, self.network, output_name_iter)
        assert trt_args is not None, \
            f'Output of {target} not exist in network.'
        return _TensorPair(torch_output, trt_args[0])
