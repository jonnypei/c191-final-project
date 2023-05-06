import numpy as np
from linear_solvers.hhl import HHL
from linear_solvers.classical_linear_solver import classical_linear_solver

A = np.array([[1, -1/3], [-1/3, 1]])
b = np.array([1, 0])

hhl_solution = HHL().solve(A, b)
print("HHL Solution:", hhl_solution)
# print(hhl_solution["state"])

classical_solution = classical_linear_solver(A, b / np.linalg.norm(b))
print("\nClassical Solution:", classical_solution)