# EMGAxO Workflow of ANN trained on MNIST Dataset


## Overview
This repository demonstrates the **EMGAxO workflow** for software level testing i-e from training a simple MNIST classifier in TensorFlow, exporting it to **ONNX**, quantizing it, replacing standard operators with **approximate QGEMM (Quantized GEMM)** operator, optimizing the quantized graph, and finally evaluating accuracy and error metrics for a full **approximate inference**.

Before running any script, make sure the environment is properly set:

## Script 1 – Training & Exporting Model

File: 01_mnist_fully_connected_model_training.py

This script trains a fully connected neural network (FCNN) on the MNIST dataset using TensorFlow and exports it as an ONNX model.

Outputs
- Trained model weights: models/best_model.h5
- Exported ONNX model: models/mnist_model.onnx


## Script 2 – Static Quantization

File: 02_mnist_static_quantization.py

This script performs static quantization of the trained FP32 ONNX model using onnxruntime.quantization. The type of Quantization is QOperator based (weights as uint8 & activations as int32).

Before running this script, make sure to preprocess the FP32 Model obained from script 1 using below command.

```bash
python3 -m onnxruntime.quantization.preprocess --input ./models/mnist_model.onnx --output ./models/mnist_model_infer.onnx
```

Outputs
- Quantized model: models/mnist_model_quantized_qgemm_uint.onnx

## Script 3 – Evaluate Original Model

File: 03_accuracy_tester_gemm.py

Evaluates the original FP32 ONNX model accuracy using EMGAxO’s check_accuracy() function.

Steps
- Loads MNIST test data (x_test.npy, y_test.npy).
- Loads and verifies ONNX model (mnist_model_infer.onnx).
- Computes accuracy, precision, recall, and F1-score.

Outputs
- Accuracy: 97.85%
- Precision: 0.9780
- Recall:    0.9791
- F1 Score:  0.9785

## Script 4 – Evaluate Quantized Model

File: 04_accuracy_tester_qgemm.py

Evaluates the quantized ONNX model performance.

Steps
- Loads mnist_model_quantized_qgemm_uint.onnx.
- Uses check_accuracy() with quantized inference enabled.
- Reports the same metrics as the FP32 model.


## Script 5 – Modify Model with APPROXIMATE QGEMM Nodes

File: 05_modifying_qgemm_with_ApproxQGemm.py

This script modifies the quantized model by replacing standard QGemm nodes with ApproxQGemm operator.

Steps

- Loads quantized model.
- Specifies target MatMul node names for replacement.
- Applies EMGAxO’s modify_model() with use_approximate_ops=True.
- Injects custom operator domain: 'test.customop'.
- Optionally sets an initial LUT configuration value (INIT_Value).

Outputs
Modified model: models/mnist_model_quantized_qgemm_uint_modified.onnx

## Script 6 – Graph Optimization

File: 06_removing_quantize_linear.py

Performs structural optimization on the quantized-approximate model using EMGAxO’s OptimizeQGraph().

Steps
- Loads modified ONNX model.
- Simplifies computation graph (constant folding, node fusions).
- Saves the optimized ONNX model.

Output
- Optimized model: models/mnist_model_quantized_qgemm_uint_optimized.onnx

## Script 7 – Evaluate Optimized Model

File: 07_accuracy_tester_ApproxQGemm.py

Evaluates the optimized QGEMM model performance using custom CUDA kernel.

Steps
- Loads optimized model.
- Runs check_accuracy() with CUDA-based custom operator support.
- Generates detailed performance metrics.

Outputs
- Accuracy: 97.42%
- Precision: 0.9741
- Recall:    0.9748
- F1 Score:  0.9744


