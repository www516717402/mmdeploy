_base_ = ['./text-recognition_dynamic.py', '../../_base_/backends/pplnn.py']

ir_config = dict(input_shape=None)

backend_config = dict(model_inputs=dict(opt_shape=[1, 1, 32, 32]))
