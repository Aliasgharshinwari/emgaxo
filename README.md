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
- [hls4ml](https://fastmachinelearning.org/hls4ml/)  
- Vitis HLS and Vivado 2022.2(for FPGA synthesis)  
- CUDA Toolkit 12.4

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aliasgharshinwari/emgaxo.git
cd emgaxo
python3 -m venv emgaxo_env
source emgaxo_env/bin/activate
pip install .
pip install -r requirements.txt

