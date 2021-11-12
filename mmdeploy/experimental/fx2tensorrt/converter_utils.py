from typing import Any, Dict, Optional, Sequence, Tuple, Union

import tensorrt as trt
import torch


def torch_dtype_to_trt(dtype: trt.DataType):
    """PyTorch dtype to TensorRT dtype."""
    if dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError(f'{dtype} is not supported by TensorRT')


def torch_dtype_from_trt(dtype):
    """TensoRT dtype to PyTorch dtype."""
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


def new_trt_const(network: trt.INetworkDefinition, tensor: Union[torch.Tensor,
                                                                 int, float]):
    """create new trt.ITensor object."""

    if isinstance(tensor, int):
        tensor = torch.tensor([tensor], dtype=torch.int32)
    elif isinstance(tensor, float):
        tensor = torch.tensor([tensor], dtype=torch.float)

    shape = tuple(tensor.shape)
    if tensor.dtype == torch.long:
        tensor = tensor.to(torch.int32)
    array = tensor.detach().cpu().numpy()
    layer = network.add_constant(shape, array)
    return layer.get_output(0)


def get_or_new_const(network: trt.INetworkDefinition,
                     tensor: Union[trt.ITensor, Any]):
    if isinstance(tensor, trt.ITensor):
        return tensor
    else:
        return new_trt_const(network, tensor)


def new_trt_const_like(network: trt.INetworkDefinition, value: Any,
                       other: Union[torch.Tensor, trt.ITensor, int, float]):
    """create new trt.ITensor object with same dtype of other."""
    other_dim = 1
    if isinstance(other, torch.Tensor):
        dtype = other.dtype
        other_dim = other.dim()
    elif isinstance(other, trt.ITensor):
        dtype = torch_dtype_from_trt(other.dtype)
        other_dim = len(other.shape)
    elif isinstance(other, int):
        dtype = torch.int32
    elif isinstance(other, float):
        dtype = torch.float
    else:
        raise TypeError(f'Can not get dtype from {type(other)}.')
    value: torch.Tensor = torch.tensor(value, dtype=dtype)
    while value.dim() < other_dim:
        value = value.unsqueeze(0)
    return new_trt_const(network, value)


def get_arg(args: Sequence,
            kwargs: Dict,
            name: str,
            pos: int,
            default: Optional[Any] = None):
    """get arguments from args and kwargs with name or pos."""
    if name in kwargs:
        return kwargs[name]
    elif len(args) > pos:
        return args[pos]
    else:
        return default


def slice_trt_shape(network: trt.INetworkDefinition,
                    shape: trt.ITensor,
                    start: int = 0,
                    size: Optional[int] = None,
                    stride: int = 1):
    """slice shape tensor in TensorRT."""
    shape_dim = shape.shape[0]
    # no need to slice
    if start == 0 and stride == 1 and (size is None or size == shape_dim):
        return shape

    if start >= shape_dim:
        return None

    if size == 0:
        return None

    if size is None:
        size = shape_dim - start

    return network.add_slice(shape, [start], [size], [stride]).get_output(0)


def get_trt_shape(network: trt.INetworkDefinition,
                  x: trt.ITensor,
                  start: int = 0,
                  size: Optional[int] = None,
                  stride: int = 1):
    """create a shape tensor from execution tensor."""
    shape = network.add_shape(x).get_output(0)
    return slice_trt_shape(network, shape, start, size, stride)


def concate_trt(network: trt.INetworkDefinition,
                *tensors: Sequence[trt.ITensor],
                dim: int = 0):
    """concatenate all tensors with given dim."""
    while dim < 0:
        dim = len(tensors[0].shape) + dim
    layer = network.add_concatenation(tensors)
    layer.axis = dim
    return layer.get_output(0)


def shuffle_with_trt_shape(network: trt.INetworkDefinition, x: trt.ITensor,
                           shape: trt.ITensor):
    """reshape x with given shape."""
    layer = network.add_shuffle(x)
    layer.set_input(1, shape)
    return layer.get_output(0)


def align_trt_dims(network: trt.INetworkDefinition, a: trt.ITensor,
                   b: trt.ITensor) -> Tuple[trt.ITensor, trt.ITensor]:
    """align the dims of a and b."""
    if len(a.shape) == len(b.shape):
        return a, b

    larger, smaller, a_is_large = (
        a, b, True) if len(a.shape) > len(b.shape) else (b, a, False)
    smaller_shape = get_trt_shape(network, smaller)

    diff = len(larger.shape) - len(smaller.shape)

    ones = new_trt_const(network, torch.ones(diff, dtype=torch.int32))
    smaller_shape = concate_trt(network, ones, smaller_shape)

    smaller = shuffle_with_trt_shape(network, smaller, smaller_shape)

    if a_is_large:
        return larger, smaller
    else:
        return smaller, larger


def cast_trt_type(network: trt.INetworkDefinition, x: trt.ITensor,
                  dtype: Union[torch.dtype, trt.DataType]):
    """cast x to given dtype."""
    if isinstance(dtype, torch.dtype):
        dtype = torch_dtype_to_trt(dtype)

    layer = network.add_identity(x)
    layer.set_output_type(0, dtype)
    out = layer.get_output(0)
    _ = out.shape  # cast might not happened if I do not add this, no idea.
    return out
