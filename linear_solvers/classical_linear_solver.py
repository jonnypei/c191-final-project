import numpy as np
from .utils.utils import generate_result

def classical_linear_solver(A, b):
    x = np.linalg.solve(A, b)
    x_norm = np.linalg.norm(x)
    
    return generate_result(x, x_norm)