import numpy as np

# 求解微分方程 dy/dx = y, y(0) = 1
h = 0.1  # 步长
x = np.arange(0, 2+h, h)  # 从0到2
y = np.zeros_like(x)
y[0] = 1  # 初值

# 欧拉法（向前差分）求解
for i in range(len(x)-1):
    y[i+1] = y[i] + h * y[i]

# 输出结果
for xi, yi in zip(x, y):
    print(f"x={xi:.1f}, y={yi:.4f}")
