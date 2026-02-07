import numpy as np

# Ax = b 形式的线性方程组
A = np.array([[1, 1, 1],
              [2, -1, 1],
              [1, 2, -1]])
b = np.array([6, 2, 4])

# 求解
x = np.linalg.solve(A, b)
print(x)  # [1. 2. 3.]