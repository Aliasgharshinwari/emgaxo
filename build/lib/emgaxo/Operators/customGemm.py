
import numpy as np
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef)

@onnx_op(op_type="CustomGemm",
                 inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
                 outputs=[PyCustomOpDef.dt_float])
def CustomGemm(A, B, C):
    """
    CustomGemm operator implementation for matrix multiplication.
    
     Parameters:
        A (np.ndarray): The matrix A.
        B (np.ndarray): The matrix B.
        C (np.ndarray): The matrix C.

    Returns:
        np.ndarray: The result of the quantized matrix multiplication.
    """

    # Get the dimensions of A and B
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape

    # Ensure that the number of columns in A is equal to the number of rows in B
    if A_cols != B_rows:
        raise ValueError("Number of columns in A must be equal to the number of rows in B.")

    # Create a result matrix initialized to zeros
    matmul_result = np.zeros((A_rows, B_cols))

    # Perform matrix multiplication and add the bias
    matmul_result = np.matmul(A, B)
    result = np.add(matmul_result, C)
    return result