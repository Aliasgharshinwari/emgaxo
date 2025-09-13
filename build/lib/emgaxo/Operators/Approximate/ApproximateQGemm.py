
import numpy as np
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef)

from emgaxo.Operators.Approximate.ApproximateMatMul import ApproximateMatMulOp

@onnx_op(op_type="ApproximateQGemm",
                 inputs=[PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8, 
                         PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8, 
                         PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float, PyCustomOpDef.dt_uint8],
                 outputs=[PyCustomOpDef.dt_uint8])
def ApproximateQGemmOp(A, A_scale, A_zeroPoint, B, B_scale, B_zeroPoint, C,  C_scale, C_zeroPoint):
    """
    QGEMM operator implementation for quantized matrix multiplication.
    
     Parameters:
        A (np.ndarray): The quantized matrix A.
        A_scale (float): The scale factor for matrix A.
        A_zeroPoint (uint8): The zero point for matrix A.
        B (np.ndarray): The quantized matrix B.
        B_scale (float): The scale factor for matrix B.
        B_zeroPoint (uint8): The zero point for matrix B.
        C (np.ndarray): The bias or accumulation matrix in int32.
        C_scale (float): The scale factor for the bias matrix C.
        C_zeroPoint (uint8): The zero point for the bias matrix C.

    Returns:
        np.ndarray: The result of the quantized matrix multiplication.
    """
    effective_scale = A_scale * B_scale
    output_scale = (effective_scale / C_scale).astype(np.float16)

    # Center A and B around their zero points
    A_centered = A.astype(np.int32) - A_zeroPoint
    B_centered = B.astype(np.int32) - B_zeroPoint

    # Perform matrix multiplication and add the bias
    matmul_result = ApproximateMatMulOp(A_centered, B_centered)
    #matmul_result = np.matmul(A_centered, B_centered)

    result = (matmul_result + C)

    # Scale the result back to uint8
    X_float = result * output_scale
    result = np.clip(X_float + C_zeroPoint, 0, 255).astype(np.uint8)

    return result