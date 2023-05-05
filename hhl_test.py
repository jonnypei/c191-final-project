import numpy as np
from linear_solvers.hhl import HHL

A = np.array([[1, -1/3], [-1/3, 1]])
b = np.array([1, 0])

naive_hhl_solution = HHL().solve(A, b)
print("Naive HHL Solution:", naive_hhl_solution)

# classical_solution = NumPyLinearSolver().solve(A, b / np.linalg.norm(b))
# print("\nClassical Solution", classical_solution)