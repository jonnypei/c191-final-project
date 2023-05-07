import numpy as np
from linear_solvers.utils.numpy_matrix import NumpyMatrix

np_mat = NumpyMatrix(np.array([[2, 0], [0, 2]]))
print(np_mat)
print(np_mat.matrix)
print(np_mat.power(3))