from scipy.optimize import fsolve
import numpy as np

# 定义方程组（将方程移项至等于0的形式）
def equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 25  # x² + y² = 25
    eq2 = x - y - 1         # x - y = 1
    return [eq1, eq2]

# 初始猜测
initial_guess = [1, 1]

# 求解
solution = fsolve(equations, initial_guess)
print(solution)  # [4. 3.]