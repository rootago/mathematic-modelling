import sympy as sp

# 符号变量
x, y, a, b = sp.symbols('x y a b')

# 方程组包含参数a, b
eq1 = x + y - a
eq2 = x - y - b

# 得到符号解（含参数的表达式）
solution = sp.solve([eq1, eq2], [x, y])
print(solution)  # {x: a/2 + b/2, y: a/2 - b/2}