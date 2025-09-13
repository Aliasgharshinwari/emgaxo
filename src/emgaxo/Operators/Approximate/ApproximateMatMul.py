import numpy as np
from emgaxo.Operators.Approximate.mult_8x8 import mult_8x8_approx

def ApproximateMatMulOp(A_centered, B_centered):
    """
    Perform approximate matrix multiplication manually between A_centered and B_centered.
    
    Parameters:
        A_centered (np.ndarray): The centered matrix A of shape (M, K).
        B_centered (np.ndarray): The centered matrix B of shape (K, N).
    
    Returns:
        np.ndarray: The result of the matrix multiplication of shape (M, N).
    """
    # Get the dimensions
    M, K = A_centered.shape
    K_B, N = B_centered.shape
    
    # Ensure the matrices are compatible for multiplication
    if K != K_B:
        raise ValueError("The number of columns in A must match the number of rows in B.")
    
    # Initialize the result matrix with zeros
    result = np.zeros((M, N), dtype=np.int32)
    
    # Perform manual matrix multiplication
    for i in range(M):
        for j in range(N):
            for k in range(K):
                result[i, j] += mult_8x8_approx(A_centered[i, k], B_centered[k, j],'acc')
                #result[i, j] += A_centered[i, k]* B_centered[k, j]
    
    return result
