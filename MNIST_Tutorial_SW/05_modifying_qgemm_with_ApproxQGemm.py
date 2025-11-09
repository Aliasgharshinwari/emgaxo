from emgaxo import modify_model

load_path = "./models/mnist_model_quantized_qgemm_uint.onnx"
save_path = "./models/mnist_model_quantized_qgemm_uint_modified.onnx"
modify_model(load_path, 
             save_path, 
             ['QGemm'], 
             [ 
            'sequential/dense/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_1/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_2/MatMul/MatMulAddFusion_quant',
            'sequential/dense_3/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_4/MatMul/MatMulAddFusion_quant',
            'sequential/dense_5/MatMul/MatMulAddFusion_quant'
            #'args_0_QuantizeLinear'
            ],
            'test.customop',
            use_approximate_ops=True,
            INIT_Value=68719476735)
