import numpy as np

# 被积函数
def f(x):
    return x**2

# 积分区间
a = 0
b = 1
N = 1000  # 分成 N 段
x = np.linspace(a, b, N+1)  # N+1 个点
dx = (b - a) / N

# 梯形法积分
I = 0
for i in range(N):
    # 每一段的梯形面积 = (f左端 + f右端)/2 * dx
    I += (f(x[i]) + f(x[i+1])) / 2 * dx

print("梯形法积分近似:", I)
