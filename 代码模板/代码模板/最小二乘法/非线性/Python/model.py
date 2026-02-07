from numpy import exp,array,inf
from scipy.optimize import least_squares

# 残差函数
def residuals(params, x, y):
    a, b, c = params
    return y - (a * exp(b * x) + c)

# 数据
x = array([0, 1, 2, 3, 4])
y = array([1.2, 2.8, 7.1, 20.1, 54.8])

# 初始猜测
initial_params = [1, 1, 1]

# 参数边界: a, b, c
# 注意：method='lm' 不能用 bounds（参数边界）
# bounds = ([0, 0, -inf], [inf, inf, inf])

# 最小二乘拟合
# result = least_squares(residuals, initial_params, args=(x, y), bounds=bounds)
# Method 参数
# trf 支持参数边界bounds
# dogbox 支持参数边界bounds
# lm 支持参数边界bounds
result = least_squares(residuals, initial_params, args=(x, y),method='lm')

print("拟合参数:", result.x)
