from emgaxo import ModifyWithAppAxO


model = ModifyWithAppAxO("./models/mnist_model_quantized_qgemm_uint.onnx", 
                               "./AppAxO_Models40k",
                                  ['QGemm'], 
             [ 
            'sequential/dense/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_1/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_2/MatMul/MatMulAddFusion_quant',
            'sequential/dense_3/MatMul/MatMulAddFusion_quant', 
            'sequential/dense_4/MatMul/MatMulAddFusion_quant',
            'sequential/dense_5/MatMul/MatMulAddFusion_quant'
            ],
            1, 10000)


