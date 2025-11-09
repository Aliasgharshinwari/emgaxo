import onnx
from emgaxo import check_accuracy
#import tensorflow as tf
import numpy as np

#(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
modified_model = onnx.load("./models/mnist_model_quantized_qgemm_uint.onnx")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

onnx.checker.check_model(modified_model)
accuracy, precision, recall, f1  = check_accuracy(modified_model, False, 'test.customop', "QGemm_Results", x_test, y_test, 10000)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
