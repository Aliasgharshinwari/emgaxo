import onnx
from emgaxo import check_accuracy
#import tensorflow as tf
import numpy as np

# === Step 1: Check and prepare MNIST test data ===
if not (os.path.exists("x_test.npy") and os.path.exists("y_test.npy")):
    print("🔄 x_test.npy or y_test.npy not found — downloading MNIST dataset...")
    import tensorflow as tf
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)
    print("✅ MNIST data downloaded and saved locally.")
else:
    print("✅ MNIST test data already exists.")


x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

model = onnx.load("./models/mnist_model_infer.onnx")
onnx.checker.check_model(model)
accuracy, precision, recall, f1 = check_accuracy(model,False,'','gemm_model_results', x_test, y_test, 10000)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
