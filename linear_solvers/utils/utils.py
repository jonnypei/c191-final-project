import numpy as np
from qiskit.quantum_info import Statevector

def check_numpy_matrix(matrix, raise_on_failure=True) -> bool:
    """Check if the current configuration is valid."""
    if matrix.shape[0] != matrix.shape[1]:
        if raise_on_failure:
            raise AttributeError("Input matrix must be square!")
        return False
    if np.log2(matrix.shape[0]) % 1 != 0:
        if raise_on_failure:
            raise AttributeError("Input matrix dimension must be 2^n!")
        return False
    if not np.allclose(matrix, matrix.conj().T):
        if raise_on_failure:
            raise AttributeError("Input matrix must be hermitian!")
        return False

    return True

def generate_result(state, euclidean_norm):
    return {"state": state,
            "euclidean_norm": euclidean_norm}
    
def get_solution_vector(solution):
    return Statevector(solution["state"]).data[16:18].real
    
