#!/usr/bin/env python3
"""
Create 1 000 approximate QGEMM variants with progressively larger
bit-flip budgets.  For each step we:

1. Take (2^36 − 1 − i)   ← 36-bit word with i low bits cleared
2. Reverse its bits
3. Use that integer as INIT_Value when calling modify_model(...)
"""

import os
from emgaxo.ModelModifier import modify_model
from emgaxo.ModelModifier import OptimizeQGraph 
from emgaxo.AccuracyTester import check_accuracy 
import onnx

WIDTH = 36
MASK  = (1 << WIDTH) - 1   # 0xFFFFFFFFF


def reverse_36bits(n: int) -> int:
    bits = f"{n:0{WIDTH}b}"
    return int(bits[::-1], 2)


def generate_reversed_sequence(start: int, end: int):
    """
    Produce reversed bit variants from i = start .. end-1:
        orig = 0xFFFFFFFFF - i
        yield reverse_36bits(orig)
    """
    for i in range(start, end):
        yield reverse_36bits(MASK - i)



def ModifyWithAppAxO(load_path: str, save_dir: str, ops_to_replace, nodes_to_replace, start: int, end: int):
    """
    • start      – starting index in the sequence
    • end        – ending index (exclusive)
    """
    os.makedirs(save_dir, exist_ok=True)

    config_ints = list(generate_reversed_sequence(start, end))

    for idx, cfg in enumerate(config_ints):
        model_save_path = os.path.join(
            save_dir,
            f"{cfg}_model.onnx",
        )

        model =    modify_model(
            source_path         = load_path,
            destination_path    = model_save_path,
            ops_to_replace      = ops_to_replace,
            nodes_to_replace    = nodes_to_replace,
            custom_domain       = "test.customop",
            use_approximate_ops = True,
            INIT_Value          = cfg,
            save_model          = False,
        )

        model = OptimizeQGraph(model, 'uint8')
        onnx.save(model, model_save_path)
    
        #accuracy = check_accuracy(model, True, 'test.customop', "QGemm_Results")


    print(f"Completed {len(config_ints)} models")
