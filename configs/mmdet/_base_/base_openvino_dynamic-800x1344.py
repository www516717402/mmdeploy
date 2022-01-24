_base_ = ['./base_dynamic.py', '../../_base_/backends/openvino.py']

ir_config = dict(input_shape=None)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))])
