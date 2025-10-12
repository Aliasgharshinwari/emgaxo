# EMGAxO Workflow on Artificial Neural Network trained on MNIST Dataset — Full Pipeline

# 🧠 EMGAxO MNIST Example Pipeline  
**Extending hls4ml with Approximate Operator Support using EMGAxO**

---

## 📘 Overview

This repository demonstrates the **complete EMGAxO workflow** — from training a simple MNIST classifier in TensorFlow, exporting it to **ONNX**, quantizing it, replacing standard operators with **approximate QGEMM (Quantized GEMM)** operators, optimizing the quantized graph, and finally evaluating accuracy and error metrics for a full **approximate inference pipeline**.

The EMGAxO library seamlessly integrates **approximate arithmetic logic** (such as LUT-based multipliers) within FPGA-accelerated or CUDA-based machine learning inference pipelines. This example uses the **MNIST** dataset for demonstration.

---

## 🧩 Table of Contents
 
1. [01 – Training & Exporting Model](#-script-1--training--exporting-model)  
2. [02 – Static Quantization](#-script-2--static-quantization)  
3. [03 – Evaluate Original Model](#-script-3--evaluate-original-model)  
4. [04 – Evaluate Quantized Model](#-script-4--evaluate-quantized-model)  
5. [05 – Modify Model with QGEMM Nodes](#-script-5--modify-model-with-qgemm-nodes)  
6. [06 – Graph Optimization](#-script-6--graph-optimization)  
7. [07 – Evaluate Optimized Model](#-script-7--evaluate-optimized-model)  
8. [08 – AppAxO Approximation Sweep](#-script-8--appaxo-approximation-sweep)  
9. [09 – Batch Model Evaluation and CSV Logging](#-script-9--batch-model-evaluation-and-csv-logging)  
10. [Output Structure](#-output-structure)

---

Before running any script, make sure the environment is properly set:

## Script 1 – Training & Exporting Model

File: train_and_export.py
This script trains a fully connected neural network (FCNN) on the MNIST dataset using TensorFlow and exports it as an ONNX model.

Outputs
- Trained model weights: models/best_model.h5
- Exported ONNX model: models/mnist_model.onnx


## Script 2 – Static Quantization
This script performs static quantization of the trained FP32 ONNX model using onnxruntime.quantization. The type of Quantization is QOperator based (weights as uint & activations as int32).

Before running this script, make sure to preprocess the FP32 Model obained from script 1 using below command.

```bash
python3 -m onnxruntime.quantization.preprocess --input ./models/mnist_model.onnx --output ./models/mnist_model_infer.onnx
```

Outputs
Quantized model: models/mnist_model_quantized_qgemm_uint.onnx

## Script 3 – Evaluate Original Model
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

## Script 4 – Evaluate Quantized Model
Evaluates the quantized ONNX model performance.

Steps
Loads mnist_model_quantized_qgemm_uint.onnx.
Uses check_accuracy() with quantized inference enabled.
Reports the same metrics as the FP32 model.


## Script 5 – Modify Model with APPROXIMATE QGEMM Nodes
This script modifies the quantized model by replacing standard QGemm nodes with ApproxQGemm operator.

Steps

Loads quantized model.
Specifies target MatMul node names for replacement.
Applies EMGAxO’s modify_model() with use_approximate_ops=True.
Injects custom operator domain: 'test.customop'.
Optionally sets an initial LUT configuration value (INIT_Value).

Outputs
Modified model: models/mnist_model_quantized_qgemm_uint_modified.onnx

## Script 6 – Graph Optimization
Performs structural optimization on the quantized-approximate model using EMGAxO’s OptimizeQGraph().

Steps
Loads modified ONNX model.
Simplifies computation graph (constant folding, node fusions).
Saves the optimized ONNX model.

Outputs
Optimized model: models/mnist_model_quantized_qgemm_uint_optimized.onnx

## Script 7 – Evaluate Optimized Model
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

## Script 8 – AppAxO Approximation Sweep
Automatically generates a set of approximate models by sweeping LUT-based approximation parameters using EMGAxO’s ModifyWithAppAxO().

Steps
Loads quantized model.
Iteratively modifies it with different 36-bit LUT configurations.
Saves multiple approximate ONNX models in a directory.

Outputs
Directory: ./AppAxO_Models40k/
Each model represents a unique approximate multiplier configuration.

## Script 9 – Batch Model Evaluation and CSV Logging

File: evaluate_all_appaxo_models.py
Evaluates all approximate models in bulk and logs results into a CSV.

Features
Automatically loads each .onnx model in a directory.
Parses configuration values from filenames (INIT integers).
Computes disabled LUT bits using 36-bit width representation.
Evaluates performance metrics and writes them to CSV.
