from numpy import array
from scipy.stats import linregress

# 数据
x = array([0, 1, 2, 3, 4])
y = array([1, 3, 7, 9, 11])

# 线性拟合
result = linregress(x, y)

print("斜率:", result.slope)
print("截距:", result.intercept)
print(f"拟合直线: y = {result.slope:.2f}x + {result.intercept:.2f}")
