# EMGAxO

**EMGAXO** (Extending Machine Learning Hardware Generators with Approximate Operators) is an **extension of [hls4ml](https://github.com/fastmachinelearning/hls4ml)** that integrates the **AppAxO Library** for employing approximate multipliers in the generated HLS Code of the FPGA based Hardware Accelerators.
It enables researchers to explore **approximate computing within FPGA-accelerated machine learning pipelines**, providing **design-space exploration** with both hardware synthesis and error metrics analysis.

---

## Overview

- Extends **hls4ml** with **approximate operator support**
- Provides Hardware Software Co Design for Efficient Approximate Hardware Accelerators for ANNs.
- Allows replacing exact multipliers with **LUT-based approximate multipliers** and showing the final accuracy of MLP Models
- Provides **error metric evaluation** before hardware implementation
- Targets FPGA deployment with **resource-accuracy trade-off exploration**  

This makes EMGAXO a powerful tool for research in **energy-efficient deep learning**, **approximate arithmetic**, and **FPGA-accelerated ML inference**.

---

## Features

- **Seamless integration** with hls4ml models
- **NVIDIA CUDA** Accelerated Software Inference for ANNs  
- **AppAxO Multipliers** 8-bit Signed Multipliers with configurable 36-bit LUT architectures  
- Built-in error analysis:  
  - Average/Absolute Error  
  - Relative Error metrics  
  - Error Probability  
  - Max/Min Error  
- **Design-space exploration** across multiple multiplier configurations  
- Support for **ONNX → hls4ml → FPGA flow** with Approximate Multipliers  

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
  url          = {https://github.com/AliasgharShinwari/emgaxo},
  version      = {1.0.0},
  institution  = {University of Engineering and Technology Peshawar, Ruhr-Universität Bochum},
  note         = {Python library for approximate computing and FPGA-based acceleration using quantized operators.}
}
```

```bash
@inproceedings{emgaxo,
  author       = {Ali Asghar and Shahzad Bangash and Suleman Shah and Salim Ullah and Laiq Hasan and Akash and Siva Satyendra Sahoo},
  title        = {{EMGAxO: Extending Machine Learning Hardware Generators with Approximate Operators}},
  booktitle    = {Proceedings of the 2025 International Conference on Compilers, Architectures, and Synthesis for Embedded Systems (CASES)},
  year         = {2025},
  organization = {University of Engineering and Technology Peshawar and Ruhr-Universität Bochum},
  url          = {https://github.com/AliasgharShinwari/emgaxo},
  note         = {A Python library for approximate computing and FPGA-based acceleration using quantized operators.},
  keywords     = {Approximate computing, FPGA, Quantized operators, Machine learning acceleration},
  doi          = {10.1109/LES.2025.3600043}
}
```
