from typing import Tuple
import numpy as np
import scipy as sp

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit.library import BlueprintCircuit
from .utils import check_numpy_matrix

class NumpyMatrix(BlueprintCircuit):
    """Class of matrices given as a numpy array."""

    def __init__(
        self,
        matrix: np.ndarray,
        tolerance: float = 1e-2,
        evolution_time: float = 1.0,
        name: str = "numpy_matrix",
    ) -> None:
        """
        Args:
            matrix: The matrix defining the linear system problem.
            tolerance: The accuracy desired for the approximation.
            evolution_time: The time of the Hamiltonian simulation.
            name: The name of the object.
        """
        
        super().__init__(name=name)

        # store parameters
        self._num_state_qubits = int(np.log2(matrix.shape[0]))
        self._reset_registers(self._num_state_qubits)
        self._tolerance = tolerance
        self._evolution_time = evolution_time  # makes sure the eigenvalues are contained in [0,1)
        self._matrix = matrix

    @property
    def num_state_qubits(self) -> int:
        """Returns the number of state qubits representing the state."""
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Sets the number of state qubits."""
        if num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @property
    def tolerance(self) -> float:
        """Returns the error tolerance."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """Sets the error tolerance."""
        self._tolerance = tolerance

    @property
    def evolution_time(self) -> float:
        """Returns the evolution time."""
        return self._evolution_time

    @evolution_time.setter
    def evolution_time(self, evolution_time: float) -> None:
        """Sets the evolution time."""
        self._evolution_time = evolution_time

    @property
    def matrix(self) -> np.ndarray:
        """Returns the matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        """Sets the matrix."""
        self._matrix = matrix

    @property
    def eigenvalue_bounds(self) -> Tuple[float, float]:
        """Returns lower and upper bounds on the eigenvalues of the matrix."""
        lambda_min = min(np.abs(np.linalg.eigvals(self.matrix)))
        lambda_max = max(np.abs(np.linalg.eigvals(self.matrix)))
        return lambda_min, lambda_max

    @property
    def condition_number(self) -> Tuple[float, float]:
        """Returns the condition number of the matrix."""
        kappa = np.linalg.cond(self.matrix)
        return kappa

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Checks if the current configuration is valid."""
        return check_numpy_matrix(matrix=self.matrix, 
                                  raise_on_failure=raise_on_failure)

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Resets the quantum registers."""
        qr_state = QuantumRegister(num_state_qubits, "state")
        self.qregs = [qr_state]

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return
        super()._build()
        self.compose(self.power(1), inplace=True)
    
    def inverse(self):
        """Returns the inverse of this matrix"""
        return NumpyMatrix(self.matrix, evolution_time=-1 * self._evolution_time)
    
    def power(self, power: int, matrix_power: bool = False) -> QuantumCircuit:
        """Build powers of the circuit.
        Args:
            power: The power to raise this circuit to.
            matrix_power: If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.
        Returns:
            The quantum circuit implementing powers of the unitary.
        """
        qc = QuantumCircuit(self._num_state_qubits)
        evolved = sp.linalg.expm(1j * self._matrix * self._evolution_time)
        qc.unitary(evolved, qc.qubits)
        return qc.power(power)
