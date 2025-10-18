# EMGAxO

**EMGAXO** (Extending Machine Learning Hardware Generators with Approximate Operators) is an **extension of [hls4ml](https://github.com/fastmachinelearning/hls4ml)** that integrates the **[AppAxO Library](https://dl.acm.org/doi/abs/10.1145/3513262)** for employing approximate multipliers in the generated HLS Code of the FPGA based Hardware Accelerators.
It enables researchers to explore **approximate computing within FPGA-accelerated machine learning pipelines**, providing **design-space exploration** with both hardware synthesis and error metrics analysis.

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

## Python Library Structure
```text
emgaxo/
├── __init__.py
├── AccuracyTester/
│   ├── __init__.py
│   ├── AccuracyTester.py               # Evaluates an ONNX model’s classification metrics (accuracy, precision, recall, F1)
│   │                                   # with optional custom operator support and GPU-based inference.
│   └── AccuracyTester_AppAxO.py        # Generates multiple ONNX model variants by progressively modifying QGEMM layers with
│                                       # different 36-bit AppAxO configurations for approximate computing exploration.
│
├── AppAxO/
│   ├── __init__.py
│   └── Evaluate_Multiplier.py          # Implements a CUDA-based LUT-configurable 8-bit approximate multiplier kernel using
│                                       # PyCUDA to compute full input pair products and evaluate detailed error metrics for
│                                       # each LUT configuration.
│
├── CustomOpLib/
│   └── libcustom_op_library.so         # Shared library of custom ONNX Runtime operators (CUDA) implementing the
│                                       # approximate quantized GEMM used by EMGAxO at inference time.
│
├── ModelModifier/
│   ├── __init__.py
│   └── ModelModifier.py               # Modifies an ONNX model by replacing target operators with custom
|                                      # approximate QGEMM nodes, optimizes the quantized graph, and rewires or updates
|                                      # tensor datatypes for approximate inference support.


```
## Installation

### Prerequisites
- Python **3.10.12**  
- CUDA Toolkit 12.4
- [hls4ml](https://fastmachinelearning.org/hls4ml/)  
- Vitis HLS and Vivado 2022.2

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

If you use **EMGAxO** in your research, please cite:

> A. Asghar *et al.*, "EMGAxO: Extending Machine Learning Hardware Generators With Approximate Operators,"  
> *IEEE Embedded Systems Letters*, vol. 17, no. 5, pp. 345–348, Oct. 2025.  
> DOI: [10.1109/LES.2025.3600043](https://doi.org/10.1109/LES.2025.3600043)

<details>
<summary>📚 BibTeX</summary>

```bibtex
@article{asghar2025emgaxo,
  author    = {Ali Asghar and Shahzad Bangash and Suleman Shah and Salim Ullah and Laiq Hasan and Akash and Siva Satyendra Sahoo},
  title     = {EMGAxO: Extending Machine Learning Hardware Generators With Approximate Operators},
  journal   = {IEEE Embedded Systems Letters},
  volume    = {17},
  number    = {5},
  pages     = {345--348},
  year      = {2025},
  doi       = {10.1109/LES.2025.3600043}
}

