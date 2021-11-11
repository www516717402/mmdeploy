import operator
from typing import Callable, Dict, List, Sequence

import pytest
import torch
from torch.fx import symbolic_trace

try:
    from mmdeploy.apis.tensorrt import TRTWrapper
    from mmdeploy.experimental.fx2tensorrt import fx2tensorrt
except ImportError:
    pytest.skip('TensorRT is not supported.', allow_module_level=True)


def _test_ops_all_close(
    torch_callable: Callable,
    inputs: List[torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    input_shapes: Dict[str, Dict[str, Sequence]],
    rtol: float = None,
    atol: float = None,
):
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    traced_model = symbolic_trace(torch_callable)
    engine = fx2tensorrt(
        traced_model,
        inputs,
        input_shapes,
        input_names=input_names,
        output_names=output_names)

    trt_model = TRTWrapper(engine)
    with torch.no_grad():
        inputs_dict = dict([(name, tensor.cuda())
                            for name, tensor in zip(input_names, inputs)])
        trt_outs = trt_model(inputs_dict)
        trt_outs = [trt_outs[name].cpu() for name in output_names]
        outs = torch_callable(*inputs)
        if isinstance(outs, torch.Tensor):
            outs = [outs]
        outs = [out.cpu() for out in outs]

    assert len(trt_outs) == len(outs)
    for trt_out, out in zip(trt_outs, outs):
        torch.testing.assert_allclose(trt_out, out)


@torch.fx.wrap
def fake_func(x):
    return x + 1


def test_default():

    x = torch.rand(2, 3, 8, 8)

    def _func_test(x):
        return x + fake_func(x) - 1

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 3, 4, 4),
            opt_shape=(2, 3, 8, 8),
            max_shape=(4, 3, 16, 16)))

    traced_model = symbolic_trace(_func_test)
    engine = fx2tensorrt(
        traced_model,
        inputs,
        input_shapes,
        input_names=input_names,
        output_names=output_names)

    trt_model = TRTWrapper(engine)
    with torch.no_grad():
        inputs_dict = dict([(name, tensor.cuda())
                            for name, tensor in zip(input_names, inputs)])
        trt_outs = trt_model(inputs_dict)
        trt_out = trt_outs['out']

        torch.testing.assert_allclose(trt_out.cpu(), 2 * x)


@pytest.mark.parametrize('elementwise_op,test_scalar',
                         [(torch.add, True), (operator.add, True),
                          (torch.sub, True), (operator.sub, True),
                          (torch.mul, True), (operator.mul, True),
                          (torch.div, True), (operator.truediv, True)])
def test_elementwise(elementwise_op, test_scalar):

    # test binary
    def _elementwise_binary_test(x, y):
        return elementwise_op(x, y)

    x = torch.rand(1, 2, 3, 4)
    y = torch.rand(2, 1, 1)
    input_names = ['x', 'y']
    output_names = ['out']
    inputs = [x, y]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 2, 1, 1),
            opt_shape=(1, 2, 3, 4),
            max_shape=(4, 2, 4, 4)),
        y=dict(min_shape=(2, 1, 1), opt_shape=(2, 1, 1), max_shape=(2, 1, 1)))

    _test_ops_all_close(
        _elementwise_binary_test,
        inputs,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes)

    if test_scalar:
        # test scalar
        def _elementwise_scalar_test(x):
            return elementwise_op(x, 5)

        input_names = ['x']
        output_names = ['out']
        inputs = [x]
        input_shapes = dict(
            x=dict(
                min_shape=(1, 2, 1, 1),
                opt_shape=(1, 2, 3, 4),
                max_shape=(4, 2, 4, 4)))
        _test_ops_all_close(
            _elementwise_scalar_test,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_conv2d():
    x = torch.rand(2, 3, 8, 8)

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, 1, 1))
    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 3, 4, 4),
            opt_shape=(2, 3, 8, 8),
            max_shape=(4, 3, 16, 16)))
    _test_ops_all_close(
        model,
        inputs,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes)


def test_getattr_shape():

    # getattr shape
    def getattr_shape(x):
        return x.shape

    x = torch.rand(1, 2, 3, 4)
    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 2, 3, 4),
            opt_shape=(2, 2, 3, 4),
            max_shape=(3, 2, 3, 4)))

    torch_callable = getattr_shape
    traced_model = symbolic_trace(torch_callable)
    engine = fx2tensorrt(
        traced_model,
        inputs,
        input_shapes,
        input_names=input_names,
        output_names=output_names)

    trt_model = TRTWrapper(engine)
    with torch.no_grad():
        inputs_dict = dict([(name, tensor.cuda())
                            for name, tensor in zip(input_names, inputs)])
        trt_outs = trt_model(inputs_dict)
        trt_outs = [trt_outs[name].cpu() for name in output_names]
        outs = torch_callable(*inputs)
        outs = [torch.tensor(outs)]

    assert len(trt_outs) == len(outs)
    for trt_out, out in zip(trt_outs, outs):
        torch.testing.assert_allclose(trt_out, out)


def test_getitem_shape():

    def getitem_shape_int(x):
        return x.shape[1]

    def getitem_shape_slice(x):
        return x.shape[1:]

    callable_list = [getitem_shape_int, getitem_shape_slice]

    x = torch.rand(1, 2, 3, 4)
    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 2, 3, 4),
            opt_shape=(2, 2, 3, 4),
            max_shape=(3, 2, 3, 4)))

    for torch_callable in callable_list:
        traced_model = symbolic_trace(torch_callable)
        engine = fx2tensorrt(
            traced_model,
            inputs,
            input_shapes,
            input_names=input_names,
            output_names=output_names)

        trt_model = TRTWrapper(engine)
        with torch.no_grad():
            inputs_dict = dict([(name, tensor.cuda())
                                for name, tensor in zip(input_names, inputs)])
            trt_outs = trt_model(inputs_dict)
            trt_outs = [trt_outs[name].cpu() for name in output_names]
            outs = torch_callable(*inputs)
            outs = [torch.tensor(outs)]

    assert len(trt_outs) == len(outs)
    for trt_out, out in zip(trt_outs, outs):
        torch.testing.assert_allclose(trt_out, out)


def test_batchnorm2d():

    model = torch.nn.Sequential(torch.nn.BatchNorm2d(8)).eval()

    x = torch.rand(2, 8, 4, 4)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))
    _test_ops_all_close(
        model,
        inputs,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes)


def test_relu():

    model = torch.nn.Sequential(torch.nn.ReLU(8)).eval()

    x = torch.rand(2, 8, 4, 4)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))
    _test_ops_all_close(
        model,
        inputs,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes)

    def _func_test(x):
        return torch.nn.functional.relu(x)

    _test_ops_all_close(
        _func_test,
        inputs,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes)


def test_max_pool2d():

    model = torch.nn.Sequential(torch.nn.MaxPool2d(2)).eval()

    def _func_test(x):
        return torch.nn.functional.max_pool2d(x, 2)

    callable_list = [model, _func_test]

    x = torch.rand(2, 8, 8, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_avg_pool2d():

    model = torch.nn.Sequential(torch.nn.AvgPool2d(2)).eval()

    def _func_test(x):
        return torch.nn.functional.avg_pool2d(x, 2)

    callable_list = [model, _func_test]

    x = torch.rand(2, 8, 8, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_adaptive_avg_pool2d():

    model1 = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1)).eval()
    model2 = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(3)).eval()

    def func1(x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 1)

    def func2(x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 3)

    callable_list = [model1, model2, func1, func2]

    x = torch.rand(2, 8, 9, 9)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(min_shape=x.shape, opt_shape=x.shape, max_shape=x.shape))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_linear():
    model = torch.nn.Sequential(torch.nn.Linear(8, 16)).eval()

    callable_list = [model]

    x = torch.rand(2, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(min_shape=(1, 8), opt_shape=(2, 8), max_shape=(4, 8)))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_flatten():

    def _func_test0(x):
        return x.flatten(1, 3)

    def _func_test1(x):
        return x.flatten(0, 2)

    def _func_test2(x):
        return x.flatten()

    callable_list = [_func_test0, _func_test1, _func_test2]

    x = torch.rand(2, 8, 8, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


@pytest.mark.parametrize('mode,align_corners', [('nearest', None),
                                                ('bilinear', False)])
def test_interpolate(mode, align_corners):

    def _resize_test(x):
        return torch.nn.functional.interpolate(
            x, size=(32, 32), mode=mode, align_corners=align_corners)

    def _scale_factor_test(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=(1.5, 1.5), mode=mode, align_corners=align_corners)

    def _resize_dynamic_test(x):
        return torch.nn.functional.interpolate(
            x,
            size=(x.shape[2] * 2, x.shape[3] * 2),
            mode=mode,
            align_corners=align_corners)

    callable_list = [_resize_test, _scale_factor_test, _resize_dynamic_test]

    x = torch.rand(2, 8, 8, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 8, 4, 4),
            opt_shape=(2, 8, 8, 8),
            max_shape=(4, 8, 16, 16)))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)


def test_size():

    # getattr shape
    def _size_test(x):
        return x.size()

    def _size_test1(x):
        return x.size(0)

    x = torch.rand(1, 2, 3, 4)
    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(
            min_shape=(1, 2, 3, 4),
            opt_shape=(2, 2, 3, 4),
            max_shape=(3, 2, 3, 4)))

    torch_callable = _size_test
    traced_model = symbolic_trace(torch_callable)
    engine = fx2tensorrt(
        traced_model,
        inputs,
        input_shapes,
        input_names=input_names,
        output_names=output_names)

    trt_model = TRTWrapper(engine)
    with torch.no_grad():
        inputs_dict = dict([(name, tensor.cuda())
                            for name, tensor in zip(input_names, inputs)])
        trt_outs = trt_model(inputs_dict)
        trt_outs = [trt_outs[name].cpu() for name in output_names]
        outs = torch_callable(*inputs)
        outs = [torch.tensor(outs)]

    assert len(trt_outs) == len(outs)
    for trt_out, out in zip(trt_outs, outs):
        torch.testing.assert_allclose(trt_out, out)

    torch_callable = _size_test1
    traced_model = symbolic_trace(torch_callable)
    engine = fx2tensorrt(
        traced_model,
        inputs,
        input_shapes,
        input_names=input_names,
        output_names=output_names)

    trt_model = TRTWrapper(engine)
    with torch.no_grad():
        inputs_dict = dict([(name, tensor.cuda())
                            for name, tensor in zip(input_names, inputs)])
        trt_outs = trt_model(inputs_dict)
        trt_outs = [trt_outs[name].cpu() for name in output_names]
        outs = torch_callable(*inputs)
        outs = [torch.tensor(outs)]

    for trt_out, out in zip(trt_outs, outs):
        torch.testing.assert_allclose(trt_out.squeeze(), out.squeeze())


def test_getitem_tensor():

    def _test_slice_static(x):
        return x[:, :, 2:4]

    def _test_slice_dynamic(x):
        return x[:, :, :x.shape[1]]

    def _test_index_select(x):
        return x[:, :, [3, 2, 1]]

    def _test_int_select(x):
        return x[:, :, 4]

    def _test_int_select_dynamic(x):
        return x[:, :, x.shape[1]]

    def _test_None(x):
        return x[:, None, ...]

    def _test_Ellipsis(x):
        return x[..., :, :2]

    callable_list = [
        _test_slice_static, _test_slice_dynamic, _test_index_select,
        _test_int_select, _test_int_select_dynamic, _test_None, _test_Ellipsis
    ]

    x = torch.rand(2, 4, 6, 8)

    input_names = ['x']
    output_names = ['out']
    inputs = [x]
    input_shapes = dict(
        x=dict(min_shape=x.shape, opt_shape=x.shape, max_shape=x.shape))

    for callable in callable_list:
        _test_ops_all_close(
            callable,
            inputs,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes)
