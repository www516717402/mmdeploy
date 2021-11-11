import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union

import tensorrt as trt
import torch
from packaging import version

from .trt_fx_parser import TRTFXParser


def fx2tensorrt(model: Union[torch.nn.Module, Callable],
                dummy_input: Any,
                input_shapes: Dict[str, Sequence[int]],
                input_names: Sequence[str],
                output_names: Sequence[str],
                log_level: trt.Logger.Severity = trt.Logger.ERROR,
                fp16_mode: bool = False,
                int8_mode: bool = False,
                int8_param: Optional[dict] = None,
                max_workspace_size: int = 0,
                device_id: int = 0,
                **kwargs):
    """convert PyTorch fx to TensorRT engine."""

    for input_name in input_names:
        assert input_name in input_shapes, \
            f'Shape info of {input_name} not provided.'
        assert len(input_shapes[input_name]) == 3, \
            f'Wrong shape info for {input_name}.'

    device = torch.device('cuda', device_id)

    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    network = builder.create_network(EXPLICIT_BATCH)

    parser = TRTFXParser(
        model,
        network,
        profile=profile,
        input_names=input_names,
        output_names=output_names)

    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    parser.run(*dummy_input)

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'
    logging.info('fx2tensorrt success.')
    return engine
