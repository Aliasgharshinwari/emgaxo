import numpy as np
from tensorflow.keras.datasets import mnist
import onnx
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat

#Run this command before running this script for the first time
#python3 -m onnxruntime.quantization.preprocess --input ./models/mnist_model.onnx --output ./models/mnist_model_infer.onnx


# 1) Load *raw* MNIST data
(x_train, _), _ = mnist.load_data()

# 2) Load your FP32 ONNX model and grab its input name
model_fp32   = "./models/mnist_model_infer.onnx"
onnx_model   = onnx.load(model_fp32)
input_name   = onnx_model.graph.input[0].name


# 3) Custom DataReader that *does* the preprocessing for you
class MnistDataReader(CalibrationDataReader):
    def __init__(self, raw_images, input_name, batch_size=1):
        """
        raw_images:  np.ndarray, shape (N, 28, 28), dtype uint8
        input_name:  the name of the ONNX graph input
        batch_size:  how many examples per batch
        """
        self.raw       = raw_images
        self.input_name = input_name
        self.batch_size = batch_size
        self.idx        = 0

    def get_next(self):
        if self.idx >= len(self.raw):
            return None

        # 3a) grab a batch of uint8 [0..255]
        batch = self.raw[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size

        # 3b) normalize to [0,1] float32
        batch = batch.astype(np.float32) / 255.0

        # 3c) flatten each 28×28 → 784
        batch = batch.reshape(batch.shape[0], 28 * 28)

        # 3d) return the dict that ORT expects
        return { self.input_name: batch }


# 4) Instantiate reader (let’s do 100 samples per batch)
reader = MnistDataReader(x_train, input_name, batch_size=100)

# 5) Run static quantization
model_quant = "./models/mnist_model_quantized_qgemm_uint.onnx"
quantize_static(
    model_fp32,
    model_quant,
    reader,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=    QuantType.QUInt8,
)

print("✅ Model successfully quantized to", model_quant)
