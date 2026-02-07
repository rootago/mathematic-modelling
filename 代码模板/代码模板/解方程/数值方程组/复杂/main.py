import sympy as sp

# 非线性方程组
x, y = sp.symbols('x y')
eq1 = x**2 + y**2 - 25
eq2 = x - y - 1

solutions = sp.solve([eq1, eq2], [x, y])
print(solutions)  # [(4, 3), (-3, -4)]