_base_ = ['./text-detection_static.py', '../../_base_/backends/onnxruntime.py']

ir_config = dict(input_shape=None)
