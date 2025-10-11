
# EMGAxO Workflow on Artificial Neural Network trained on MNIST Dataset — Full Pipeline

This folder demonstrates a complete **end-to-end workflow** using the **EMGAxO** library for integrating approximate multiplier from AppAxO library into the ANN trained on MNIST Dataset.  
It trains, quantizes, modifies, and evaluates a **fully connected MNIST classifier** through the following stages:

1. TensorFlow model training  
2. ONNX export and quantization  
3. QGEMM custom operator integration  
4. Approximation and optimization  
5. Accuracy evaluation and batch testing  

---

## 🧭 Table of Contents

1. [Overview](#overview)  
2. [Setup Instructions](#setup-instructions)  
3. [Script 1 — Train and Export Model](#script-1--train-and-export-model)  
4. [Script 2 — Quantize the Model](#script-2--quantize-the-model)  
5. [Script 3 — Evaluate Original Model](#script-3--evaluate-original-model)  
6. [Script 4 — Evaluate Quantized Model](#script-4--evaluate-quantized-model)  
7. [Script 5 — Replace GEMM with QGEMM](#script-5--replace-gemm-with-qgemm)  
8. [Script 6 — Optimize Quantized Graph](#script-6--optimize-quantized-graph)  
9. [Script 7 — Evaluate Optimized Approximate Model](#script-7--evaluate-optimized-approximate-model)  
10. [Script 8 — Generate Multiple Approximate Models](#script-8--generate-multiple-approximate-models)  
11. [Script 9 — Evaluate All Approximate Models in Bulk](#script-9--evaluate-all-approximate-models-in-bulk)  
---

## 🧩 Overview
The **EMGAxO** framework provides a platform for integrating **approximate computing operators** into ONNX-based quantized models.  
This workflow focuses on the **MNIST digit classification** task using a dense neural network trained in TensorFlow.
---
