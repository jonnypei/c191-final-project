from typing import Dict
import numpy as np
from .utils.utils import generate_result

def classical_linear_solver(A, b) -> Dict:
    x = np.linalg.solve(A, b)
    x_norm = np.linalg.norm(x)
    
    return generate_result(x.tolist(), x_norm)