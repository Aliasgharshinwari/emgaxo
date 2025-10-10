import onnx
import os
from emgaxo import OptimizeQGraph 

model_path = "./models/mnist_model_quantized_qgemm_uint_modified.onnx"
optimized_model_path = "./models/mnist_model_quantized_qgemm_uint_optimized.onnx"

#model_path = "./models/fcnn_quantized_modified.onnx"
#optimized_model_path = "./models/fcnn_quantized_modified_optimized.onnx"

# Ensure the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File not found: {model_path}")

model = onnx.load(model_path)
modified_model = OptimizeQGraph(model, 'uint8')
onnx.save(modified_model, optimized_model_path)
