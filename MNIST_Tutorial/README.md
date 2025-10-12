
# EMGAxO Workflow on Artificial Neural Network trained on MNIST Dataset — Full Pipeline

🧮 Script 1 – Training & Exporting Model

File: train_and_export.py

This script trains a fully connected neural network (FCNN) on the MNIST dataset using TensorFlow and exports it as an ONNX model.

Features

GPU configuration with memory growth handling.

Logging with timestamps and severity levels.

Early stopping, learning rate reduction, and checkpointing.

Automatic ONNX export via tf2onnx.

Outputs

Trained model weights: models/best_model.h5

Exported ONNX model: models/mnist_model.onnx

Command
python train_and_export.py

🧭 Script 2 – Static Quantization

File: quantize_mnist_model.py

This script performs static quantization of the trained FP32 ONNX model using onnxruntime.quantization.

Steps

Loads MNIST data (only inputs, not labels).

Preprocesses the ONNX model for inference:

python3 -m onnxruntime.quantization.preprocess \
--input ./models/mnist_model.onnx \
--output ./models/mnist_model_infer.onnx


Defines a CalibrationDataReader that normalizes and flattens MNIST data.

Performs QOperator quantization (both weights & activations as uint8).

Outputs

Quantized model: models/mnist_model_quantized_qgemm_uint.onnx

Command
python quantize_mnist_model.py

🧪 Script 3 – Evaluate Original Model

File: evaluate_original_onnx.py

Evaluates the original FP32 ONNX model accuracy using EMGAxO’s check_accuracy() function.

Steps

Loads MNIST test data (x_test.npy, y_test.npy).

Loads and verifies ONNX model (mnist_model_infer.onnx).

Computes accuracy, precision, recall, and F1-score.

Outputs
Accuracy: 97.85%
Precision: 0.9780
Recall:    0.9791
F1 Score:  0.9785

⚡ Script 4 – Evaluate Quantized Model

File: evaluate_quantized_model.py

Evaluates the quantized ONNX model performance.

Steps

Loads mnist_model_quantized_qgemm_uint.onnx.

Uses check_accuracy() with quantized inference enabled.

Reports the same metrics as the FP32 model.

Command
python evaluate_quantized_model.py

🔧 Script 5 – Modify Model with QGEMM Nodes

File: modify_with_qgemm.py

This script modifies the quantized model by replacing standard matrix multiplication nodes with QGemm (Quantized GEMM) custom operators.

Steps

Loads quantized model.

Specifies target MatMul node names for replacement.

Applies EMGAxO’s modify_model() with use_approximate_ops=True.

Injects custom operator domain: 'test.customop'.

Optionally sets an initial LUT configuration value (INIT_Value).

Outputs

Modified model: models/mnist_model_quantized_qgemm_uint_modified.onnx

🧠 Script 6 – Graph Optimization

File: optimize_qgemm_graph.py

Performs structural optimization on the quantized-approximate model using EMGAxO’s OptimizeQGraph().

Steps

Loads modified ONNX model.

Simplifies computation graph (constant folding, node fusions).

Saves the optimized ONNX model.

Outputs

Optimized model: models/mnist_model_quantized_qgemm_uint_optimized.onnx

📊 Script 7 – Evaluate Optimized Model

File: evaluate_optimized_model.py

Evaluates the optimized QGEMM model performance using custom CUDA kernels.

Steps

Loads optimized model.

Runs check_accuracy() with CUDA-based custom operator support.

Generates detailed performance metrics.

Outputs
Accuracy: 97.42%
Precision: 0.9741
Recall:    0.9748
F1 Score:  0.9744

🧩 Script 8 – AppAxO Approximation Sweep

File: generate_appaxo_models.py

Automatically generates a set of approximate models by sweeping LUT-based approximation parameters using EMGAxO’s ModifyWithAppAxO().

Steps

Loads quantized model.

Iteratively modifies it with different 36-bit LUT configurations.

Saves multiple approximate ONNX models in a directory.

Outputs

Directory: ./AppAxO_Models40k/
Each model represents a unique approximate multiplier configuration.

📈 Script 9 – Batch Model Evaluation and CSV Logging

File: evaluate_all_appaxo_models.py

Evaluates all approximate models in bulk and logs results into a CSV.

Features

Automatically loads each .onnx model in a directory.

Parses configuration values from filenames (INIT integers).

Computes disabled LUT bits using 36-bit width representation.

Evaluates performance metrics and writes them to CSV.
