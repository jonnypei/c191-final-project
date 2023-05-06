from typing import Union, List, Dict
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.opflow import (
    Z,
    I,
    StateFn,
    TensoredOp,
)

from .utils.numpy_matrix import NumpyMatrix 
from .utils.utils import check_numpy_matrix, generate_result
    
class HHL():
    """The HHL Algorithm"""

    def __init__(self, epsilon=1e-2, scaling=1):
        # Tolerance param
        self._epsilon = epsilon
        
        # Tolerances for each part of algorithm
        self._epsilon_r = epsilon / 3  # conditioned rotation
        self._epsilon_a = epsilon / 6  # hamiltonian simulation

        # Scaling Factor for eigenvalue representation
        self._scaling = scaling

    def get_delta(self, n_lambda: int, lambda_min: float, lambda_max: float) -> float:
        """Computes the scaling factor to represent lambda_min exactly using n_lambda binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: The smallest eigenvalue of the system.
            lambda_max: The largest eigenvalue of the system.

        Returns:
            The value of the scaling factor.
        """
        lambda_min_tilde = np.abs(lambda_min * (2**n_lambda - 1) / lambda_max)
        # correct floating point precision issues
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
            
        lambda_min_binary = format(int(lambda_min_tilde), 
                                   "#0" + str(n_lambda + 2) + "b")[2::]
        
        lambda_min_rep = 0
        for i, bit in enumerate(lambda_min_binary):
            lambda_min_rep += int(bit) / (2 ** (i + 1))
        return lambda_min_rep

    def calculate_norm(self, qc: QuantumCircuit) -> float:
        """Calculates the euclidean norm of the solution.

        Args:
            qc: The quantum circuit preparing the solution x to the system.

        Returns:
            The value of the euclidean norm of the solution.
        """
        # Calculate the number of qubits
        n_b = qc.qregs[0].size      # number of qubits for b (in Ax=b)
        n_lambda = qc.qregs[1].size # number of eigenvalue evaluation qubits
        n_ancilla = qc.num_ancillas # number of ancilla qubits

        # Create zero and one operators
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        # Norm observable
        M = one_op ^ TensoredOp((n_lambda + n_ancilla) * [zero_op]) ^ (I ^ n_b)
        norm_2 = (~StateFn(M) @ StateFn(qc)).eval()

        return np.real(np.sqrt(norm_2) / self._scaling)

    def construct_circuit(self, 
                          matrix: Union[List, np.ndarray], 
                          vector: Union[List, np.ndarray], 
                          neg_vals=True
                          ) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix of the system, i.e. A in Ax=b
            vector: The vector of the system, i.e. b in Ax=b
            neg_vals: States whether the matrix has negative eigenvalues.
        
        Raises:
            ValueError: If the input is not in the correct format.
            ValueError: If the type of the input matrix is not supported.

        Returns:
            The HHL circuit.
        """
        # State preparation circuit
        assert isinstance(vector, (list, np.ndarray)), f"Invalid type for vector: {type(vector)}."
        if isinstance(vector, list):
            vector = np.array(vector)
        n_b = int(np.log2(len(vector)))
        vector_circuit = QuantumCircuit(n_b)
        vector_circuit.isometry(vector / np.linalg.norm(vector), list(range(n_b)), None)

        # State preparation is probabilistic the number of qubit flags should increase
        n_flag = 1

        # Hamiltonian simulation circuit (Trotterization is default, though we do not implement it here)
        assert isinstance(matrix, (list, np.ndarray)), f"Invalid type for matrix: {type(matrix)}."
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        check_numpy_matrix(matrix)
        if matrix.shape[0] != 2**vector_circuit.num_qubits:
            raise ValueError(
                "Input vector dimension does not match input "
                "matrix dimension! Vector dimension: "
                + str(vector_circuit.num_qubits)
                + ". Matrix dimension: "
                + str(matrix.shape[0])
            )
        matrix_circuit = NumpyMatrix(matrix, evolution_time=2 * np.pi)

        # Set the tolerance for the matrix approximation
        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        # Compute the condition number
        if matrix_circuit.condition_number is not None:
            kappa = matrix_circuit.condition_number
        else:
            kappa = 1
            
        # Update the number of qubits required to represent the eigenvalues
        # The +neg_vals is to register negative eigenvalues because
        # e^{-2 \pi i \lambda} = e^{2 \pi i (1 - \lambda)}
        n_lambda = max(n_b + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        # Compute bounds for the eigenvalues of the system
        if matrix_circuit.eigs_bounds is not None:
            lambda_min, lambda_max = matrix_circuit.eigs_bounds
            
            # Constant so that the minimum eigenvalue is represented exactly; 
            # if there are negative eigenvalues, -1 to take into account the sign qubit
            delta = self.get_delta(n_lambda - neg_vals, lambda_min, lambda_max)
            # Update evolution time
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            # Update the scaling of the solution
            self._scaling = lambda_min
        else:
            delta = 1 / (2**n_lambda)
            print("Note: the solution will be calculated up to a scaling factor.")

        reciprocal_circuit = ExactReciprocal(n_lambda, delta, neg_vals=neg_vals)
        # Update number of ancilla qubits
        n_ancilla = matrix_circuit.num_ancillas
        
        # Initialise the quantum registers
        q_b = QuantumRegister(n_b)                  # initially stores b, ultimately stores solution
        q_lambda = QuantumRegister(n_lambda)        # eigenvalue evaluation qubits
        if n_ancilla > 0:
            q_ancilla = AncillaRegister(n_ancilla)  # ancilla qubits
        q_flag = QuantumRegister(n_flag)            # flag qubits

        if n_ancilla > 0:
            qc = QuantumCircuit(q_b, q_lambda, q_ancilla, q_flag)
        else:
            qc = QuantumCircuit(q_b, q_lambda, q_flag)

        # State preparation
        qc.append(vector_circuit, q_b[:])
        
        # QPE
        phase_estimation = PhaseEstimation(n_lambda, matrix_circuit)
        if n_ancilla > 0:
            qc.append(phase_estimation, q_lambda[:] + q_b[:] + q_ancilla[: matrix_circuit.num_ancillas])
        else:
            qc.append(phase_estimation, q_lambda[:] + q_b[:])
            
        # Conditioned rotation
        qc.append(reciprocal_circuit, q_lambda[::-1] + [q_flag[0]])
        
        # Inverse QPE
        if n_ancilla > 0:
            qc.append(
                phase_estimation.inverse(),
                q_lambda[:] + q_b[:] + q_ancilla[: matrix_circuit.num_ancillas]
            )
        else:
            qc.append(phase_estimation.inverse(), q_lambda[:] + q_b[:])
            
        return qc

    def solve(self, 
              matrix: Union[List, np.ndarray], 
              vector: Union[List, np.ndarray]
              ) -> Dict:
        """Solves for ||x||_2 in the given linear system of equations: Ax=b.

        Args:
            matrix: The matrix of the system, i.e. A in Ax=b
            vector: The vector of the system, i.e. b in Ax=b

        Returns:
            The L2 norm of the solution of the system, i.e. ||x||_2 where x is from Ax=b
        """
        state = self.construct_circuit(matrix, vector)
        euclidean_norm = self.calculate_norm(state)

        return generate_result(state, euclidean_norm)
