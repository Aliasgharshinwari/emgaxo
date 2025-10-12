# EMGAxO

**EMGAXO** (Extending Machine Learning Hardware Generators with Approximate Operators) is an **extension of [hls4ml](https://github.com/fastmachinelearning/hls4ml)** that integrates the **[AppAxO Library](https://dl.acm.org/doi/abs/10.1145/3513262)** for employing approximate multipliers in the generated HLS Code of the FPGA based Hardware Accelerators.
It enables researchers to explore **approximate computing within FPGA-accelerated machine learning pipelines**, providing **design-space exploration** with both hardware synthesis and error metrics analysis.

---

tree 
## Structure
emgaxo/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── builder.py               # Handles HLS model generation and integration with hls4ml
│   ├── converter.py             # Converts ONNX → hls4ml and injects approximate operators
│   ├── evaluator.py             # Evaluates accuracy, latency, and error metrics
│   ├── explorer.py              # Performs design-space exploration (resource vs accuracy)
│   └── utils.py                 # Common utility functions for data handling and logging
│
├── approx/
│   ├── __init__.py
│   ├── appaxo_lut.py            # AppAxO-based LUT multiplier implementations
│   ├── quantizer.py             # Quantization and dequantization helpers
│   ├── error_metrics.py         # Computes Average, Relative, and Max Error metrics
│   └── config.py                # Configuration for approximate multiplier selection
│
├── cuda/
│   ├── __init__.py
│   ├── qgemm_cuda.py            # CUDA implementation of approximate quantized GEMM
│   ├── kernels/                 # Custom CUDA kernels for inference
│   │   ├── lut_mul_kernel.cu    # Low-level kernel for LUT-based approximate multiplication
│   │   └── ...
│   └── runtime.py               # Manages CUDA execution and memory transfer
│
├── hls/
│   ├── __init__.py
│   ├── templates/               # HDL/HLS code templates used for code generation
│   ├── hls_generator.py         # Generates synthesizable C++/HLS code for FPGA
│   └── vivado_utils.py          # Vivado/Vitis integration scripts
│
├── examples/
│   ├── MNIST_Tutorial/          # End-to-end example for MNIST using approximate MLP
│   ├── CIFAR_Tutorial/          # (Optional) CIFAR-10 example
│   └── simple_demo.py           # Quick-start demo for testing multipliers
│
├── tests/
│   ├── test_lut_multiplier.py   # Unit tests for AppAxO LUT multipliers
│   ├── test_qgemm_cuda.py       # Tests for CUDA quantized GEMM
│   ├── test_converter.py        # Verifies ONNX → hls4ml flow correctness
│   └── test_metrics.py          # Validates error metric computations
│
├── requirements.txt
├── setup.py
└── README.md

---
## Features

- Extends **hls4ml** with **approximate operator support**
- **Seamless integration** with hls4ml models
- **NVIDIA CUDA** Accelerated Software Inference for ANNs  
- **AppAxO's** 8-bit Signed Multipliers with configurable 36-bit LUT architectures
- Allows replacing exact multipliers with **LUT-based approximate multipliers** and showing the final accuracy of MLP Models
- Built-in error analysis:  
  - Average/Absolute Error  
  - Relative Error metrics  
  - Error Probability  
  - Max/Min Error  
- **Design-space exploration** across multiple multiplier configurations  
- Support for **ONNX → hls4ml → FPGA flow** with Approximate Multipliers  
- Targets FPGA deployment with **resource-accuracy trade-off exploration**  

---

## Installation

### Prerequisites

- Python **3.10.12**  
- CUDA Toolkit 12.4
- [hls4ml](https://fastmachinelearning.org/hls4ml/)  
- Vitis HLS and Vivado 2022.2(for FPGA synthesis)  

### Step 1: Clone the Repository
```bash
git clone https://github.com/Aliasgharshinwari/emgaxo.git
cd emgaxo
```

### Step 2: Create and Activate Virtual Environment for Package Dependancies
```bash
python3 -m venv emgaxo_env
source emgaxo_env/bin/activate
```

### Step 3: Install the EMGAxO Python Library as well as the required dependancies
```bash
pip install .
pip install -r requirements.txt
```

### Step 4: Force Install OnnxRuntime GPU Package for GPU based Inference
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.21.0
```

### Step 5: Run the Example Scripts
Head over to the MNIST_Tutorial Folder and run the example scripts in their respective order.


## Citation
If you use this library in a publication, please cite the software/article
```bash
@software{emgaxo,
  author       = {Ali Asghar, Shahzad Bangash, Suleman Shah, Dr. Salim Ullah, Dr. Laiq Hasan, Dr. Akash, Dr. Siva Satyendra Sahoo},
  title        = {EMGAxO: Extending Machine Learning Hardware Generators with Approximate Operators},
  year         = {2025},
  url          = {https://github.com/Aliasgharshinwari/emgaxo},
  version      = {1.0.0},
  institution  = {University of Engineering and Technology Peshawar and Ruhr-Universität Bochum},
  note         = {EMGAxO extends hls4ml with the AppAxO Library to enable approximate computing in FPGA-based machine learning accelerators.}
}
```

```bash
@inproceedings{emgaxo,
  author       = {Ali Asghar and Shahzad Bangash and Suleman Shah and Salim Ullah and Laiq Hasan and Akash and Siva Satyendra Sahoo},
  title        = {{EMGAxO: Extending Machine Learning Hardware Generators with Approximate Operators}},
  booktitle    = {Proceedings of the 2025 International Conference on Compilers, Architectures, and Synthesis for Embedded Systems (CASES)},
  year         = {2025},
  organization = {University of Engineering and Technology Peshawar and Ruhr-Universität Bochum},
  url          = {https://github.com/Aliasgharshinwari/emgaxo},
  note         = {EMGAxO extends hls4ml with the AppAxO Library to enable approximate computing in FPGA-based machine learning accelerators.},
  keywords     = {Approximate computing, FPGA, Quantized operators, Machine learning acceleration},
  doi          = {10.1109/LES.2025.3600043}
}
```
