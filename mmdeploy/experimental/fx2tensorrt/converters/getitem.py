# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# modified from
# https://github.com/NVIDIA-AI-IOT/torch2trt
from typing import Any, Dict, Sequence, Tuple

import tensorrt as trt
import torch

from ..converter_registry import TRT_REGISTRY
from ..converter_utils import get_trt_shape, new_trt_const, slice_trt_shape


def __convert__getitem__sequence(ctx: Any, torch_args: Tuple[Any, ...],
                                 torch_kwargs: Dict[str, Any],
                                 trt_args: Tuple[Any,
                                                 ...], trt_kwargs: Dict[str,
                                                                        Any]):
    seq = trt_args[0]
    slices = trt_args[1]

    return seq[slices]


def __convert__getitem__torchsize(ctx: Any, torch_args: Tuple[Any, ...],
                                  torch_kwargs: Dict[str, Any],
                                  trt_args: Tuple[Any,
                                                  ...], trt_kwargs: Dict[str,
                                                                         Any]):
    torch_x = torch_args[0]
    trt_x = trt_args[0]

    index = torch_args[1]
    torch_dim = len(torch_x)

    # process Size
    if isinstance(index, int):
        if index < 0:
            index = index + torch_dim
        return slice_trt_shape(ctx.network, trt_x, index, 1)
    elif isinstance(index, slice):
        shape_slice = index
        start = 0 if shape_slice.start is None else shape_slice.start
        start = start if start >= 0 else start + torch_dim
        stop = len(
            torch_args[0]) if shape_slice.stop is None else shape_slice.stop
        stop = stop if stop >= 0 else stop + torch_dim
        size = stop - start
        step = 1 if shape_slice.step is None else shape_slice.step

        return slice_trt_shape(
            ctx.network, trt_x, start=start, size=size, stride=step)
    else:
        raise ValueError(f'Unknown getitem type {type(index)}')


def __num_slice_or_gather(slices: Sequence):
    return sum([1 for s in slices if isinstance(s, (slice, int, Sequence))])


def __parse_slice(network: trt.INetworkDefinition, torch_slice: slice,
                  trt_slice: slice, torch_dim_size: int,
                  trt_dim_size: trt.ITensor):
    torch_start = 0 if torch_slice.start is None else torch_slice.start
    torch_stop = torch_dim_size if torch_slice.stop is None\
        else torch_slice.stop
    torch_stride = 1 if torch_slice.step is None else torch_slice.step
    torch_size = (torch_stop - torch_start - 1) // torch_stride + 1

    trt_start = 0 if trt_slice.start is None else trt_slice.start
    trt_stop = trt_dim_size if trt_slice.stop is None else trt_slice.stop
    trt_stride = 1 if trt_slice.step is None else trt_slice.step

    need_dynamic = False

    # dynamic if start <0 or stop < 0
    if torch_start < 0 or torch_stop < 0:
        need_dynamic = True

    # dynamic if stop is None
    if torch_slice.stop is None:
        need_dynamic = True

    if isinstance(trt_start, trt.ITensor) or isinstance(
            trt_stop, trt.ITensor) or isinstance(trt_stride, trt.ITensor):
        need_dynamic = True

    if not isinstance(trt_start, trt.ITensor):
        trt_start = new_trt_const(network,
                                  torch.tensor([trt_start], dtype=torch.int32))
    if not isinstance(trt_stop, trt.ITensor):
        trt_stop = new_trt_const(network,
                                 torch.tensor([trt_stop], dtype=torch.int32))
    if not isinstance(trt_stride, trt.ITensor):
        trt_stride = new_trt_const(
            network, torch.tensor([trt_stride], dtype=torch.int32))

    if not need_dynamic:
        trt_size = new_trt_const(network,
                                 torch.tensor([torch_size], dtype=torch.int32))
    else:
        if torch_start < 0:
            torch_start = torch_start + torch_dim_size
            trt_start = network.add_elementwise(
                trt_dim_size, trt_start,
                trt.ElementWiseOperation.SUM).get_output(0)

        if torch_stop < 0:
            torch_stop = torch_stop + torch_dim_size
            trt_stop = network.add_elementsize(
                trt_dim_size, trt_stop,
                trt.ElementWiseOperation.SUM).get_output(0)

        trt_one = new_trt_const(network, 1)
        # stop - start
        trt_size = network.add_elementwise(
            trt_stop, trt_start, trt.ElementWiseOperation.SUB).get_output(0)
        # stop - start - 1
        trt_size = network.add_elementwise(
            trt_size, trt_one, trt.ElementWiseOperation.SUB).get_output(0)
        # (stop - start - 1) // stride
        trt_size = network.add_elementwise(
            trt_size, trt_stride, trt.ElementWiseOperation.DIV).get_output(0)
        # (stop - start - 1) // stride + 1
        trt_size = network.add_elementwise(
            trt_size, trt_one, trt.ElementWiseOperation.SUM).get_output(0)

    return need_dynamic, (torch_start, torch_size,
                          torch_stride), (trt_start, trt_size, trt_stride)


def __convert__getitem__tensor(ctx: Any, torch_args: Tuple[Any, ...],
                               torch_kwargs: Dict[str, Any],
                               trt_args: Tuple[Any,
                                               ...], trt_kwargs: Dict[str,
                                                                      Any]):
    torch_input = torch_args[0]
    torch_slices = torch_args[1]
    trt_input = trt_args[0]
    trt_slices = trt_args[1]

    if not isinstance(trt_slices, tuple):
        torch_slices = (torch_slices, )
        trt_slices = (trt_slices, )

    # choice what to do
    num_slice_or_gather = __num_slice_or_gather(trt_slices)
    num_ellipsis = torch_input.dim() - num_slice_or_gather

    # slice/int
    new_torch_slices = []
    new_trt_slices = []
    # tensor
    new_gathers = []
    # int
    erase_dims = []
    # None
    add_dims = []

    ellipsis_count = 0
    for index, (torch_s, trt_s) in enumerate(zip(torch_slices, trt_slices)):
        if torch_s is Ellipsis:
            for _ in range(num_ellipsis):
                new_torch_slices.append(slice(None, None, None))
                new_trt_slices.append(slice(None, None, None))
                new_gathers.append(None)
            ellipsis_count = num_ellipsis - 1
        elif isinstance(torch_s, slice):
            new_torch_slices.append(torch_s)
            new_trt_slices.append(trt_s)
            new_gathers.append(None)
        elif torch_s is None:
            add_dims.append(index + ellipsis_count)
        elif isinstance(torch_s, int):
            erase_dims.append(index + ellipsis_count)
            new_torch_slices.append(torch_s)
            new_trt_slices.append(trt_s)
            new_gathers.append(None)
        elif isinstance(torch_s, Sequence):
            new_torch_slices.append(slice(None, None, None))
            new_trt_slices.append(slice(None, None, None))
            new_gathers.append(trt_s)
        elif isinstance(torch_s, torch.Tensor) and torch_s.dtype == torch.bool:
            raise TypeError('Does not support getitem with binary mask.')

    # fill missing slices at end
    while __num_slice_or_gather(new_trt_slices) < torch_input.dim():
        new_torch_slices.append(slice(None, None, None))
        new_trt_slices.append(slice(None, None, None))
        new_gathers.append(None)

    need_slice = any(
        [trt_s != slice(None, None, None) for trt_s in new_trt_slices])
    need_gather = any([g is not None for g in new_gathers])

    trt_one = new_trt_const(ctx.network, 1)
    trt_output = trt_input
    if need_slice:
        # slice tensor
        # for static slice
        torch_starts = []
        torch_sizes = []
        torch_strides = []

        # for dynamic slice
        trt_starts = []
        trt_sizes = []
        trt_strides = []

        need_dynamic_input = False
        for index, (torch_s,
                    trt_s) in enumerate(zip(new_torch_slices, new_trt_slices)):
            if index >= torch_input.dim():
                break

            trt_dim_size = get_trt_shape(ctx.network, trt_input, index, 1)
            torch_dim_size = int(torch_input.shape[index])

            if isinstance(torch_s, slice):
                need_dyna, torch_res, trt_res = __parse_slice(
                    ctx.network,
                    torch_s,
                    trt_s,
                    torch_dim_size=torch_dim_size,
                    trt_dim_size=trt_dim_size)
                need_dynamic_input = need_dynamic_input or need_dyna
                torch_starts.append(torch_res[0])
                torch_sizes.append(torch_res[1])
                torch_strides.append(torch_res[2])
                trt_starts.append(trt_res[0])
                trt_sizes.append(trt_res[1])
                trt_strides.append(trt_res[2])
            elif isinstance(torch_s, int):
                torch_starts.append(torch_s)
                torch_sizes.append(1)
                torch_strides.append(1)

                if not isinstance(trt_s, trt.ITensor):
                    trt_s = new_trt_const(ctx.network, trt_s)
                else:
                    need_dynamic_input = True

                if torch_s < 0:
                    need_dynamic_input = True
                    trt_s = ctx.network.add_elementwise(
                        trt_s, trt_dim_size,
                        trt.ElementWiseOperation.SUM).get_output(0)

                trt_starts.append(trt_s)
                trt_sizes.append(trt_one)
                trt_strides.append(trt_one)

        if not need_dynamic_input:
            trt_output = ctx.network.add_slice(trt_input, torch_starts,
                                               torch_sizes,
                                               torch_strides).get_output(0)
        else:
            trt_starts = ctx.network.add_concatenation(trt_starts).get_output(
                0)
            trt_sizes = ctx.network.add_concatenation(trt_sizes).get_output(0)
            trt_strides = ctx.network.add_concatenation(
                trt_strides).get_output(0)
            slice_layer = ctx.network.add_slice(trt_input, torch_starts,
                                                torch_sizes, torch_strides)
            slice_layer.set_input(1, trt_starts)
            slice_layer.set_input(2, trt_sizes)
            slice_layer.set_input(3, trt_strides)
            trt_output = slice_layer.get_output(0)

    if need_gather:
        # gather
        for gidx, gather_value in enumerate(new_gathers):
            if gather_value is None:
                continue

            if not isinstance(gather_value, trt.ITensor):
                gather_value = new_trt_const(
                    ctx.network, torch.tensor(gather_value, dtype=torch.int32))
            trt_output = ctx.network.add_gather(trt_output, gather_value,
                                                gidx).get_output(0)

    # add shuffle layer if necessary
    if len(erase_dims) + len(add_dims) > 0:
        layer = ctx.network.add_shuffle(trt_output)
        # full output shape
        trt_out_shape = [
            get_trt_shape(ctx.network, trt_output, i, 1)
            for i in range(torch_input.dim())
        ]

        # if slice is None
        for add_d in add_dims[::-1]:
            trt_out_shape = trt_out_shape[:add_d] + [trt_one
                                                     ] + trt_out_shape[add_d:]

        # if slice is Int
        for e in erase_dims:
            trt_out_shape[e] = None
        trt_out_shape = list(filter(lambda x: x is not None, trt_out_shape))
        if len(trt_out_shape) > 0:
            trt_out_shape = ctx.network.add_concatenation(
                trt_out_shape).get_output(0)
            layer.set_input(1, trt_out_shape)
        else:
            layer.reshape_dims = (1, )

        trt_output = layer.get_output(0)

    return trt_output


@TRT_REGISTRY.register_converter('operator.getitem')
def convert__getitem(ctx: Any, torch_args: Tuple[Any, ...],
                     torch_kwargs: Dict[str, Any], trt_args: Tuple[Any, ...],
                     trt_kwargs: Dict[str, Any], **kwargs):
    """convert getitem to tensorrt."""
    x = torch_args[0]

    if isinstance(x, torch.Size):
        return __convert__getitem__torchsize(ctx, torch_args, torch_kwargs,
                                             trt_args, trt_kwargs)
    elif isinstance(x, torch.Tensor):
        return __convert__getitem__tensor(ctx, torch_args, torch_kwargs,
                                          trt_args, trt_kwargs)
    elif isinstance(x, Sequence):
        return __convert__getitem__sequence(ctx, torch_args, torch_kwargs,
                                            trt_args, trt_kwargs)
    else:
        raise TypeError(f'Unknown getitem of {type(x)}.')
