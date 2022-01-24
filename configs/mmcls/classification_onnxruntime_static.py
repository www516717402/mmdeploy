_base_ = ['./classification_static.py', '../_base_/backends/onnxruntime.py']

ir_config = dict(input_shape=None)
