import numpy as np
from linear_solvers.hhl import HHL
from linear_solvers.classical_linear_solver import classical_linear_solver
from linear_solvers.utils.utils import get_solution_vector
 
A = np.array([[1, 2], [2, 1]])
b = np.array([1, 0])

hhl_solver = HHL()
hhl_solution = hhl_solver.solve(A, b)
print("HHL Solution:", hhl_solution)
print("HHL Solution Vector:", get_solution_vector(hhl_solution))
# print(hhl_solution["state"])

classical_solution = classical_linear_solver(A, b / np.linalg.norm(b))
print("\nClassical Solution:", classical_solution)