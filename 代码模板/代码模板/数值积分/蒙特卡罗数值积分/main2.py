from numpy import random, mean, prod

# 被积函数
def f(x, y):
    return x**2 + y**2

# 积分区间
bounds = [(0, 1), (0, 1)]  # [(a1,b1), (a2,b2)]
N = 100000

# 在每个维度上采样
x = random.uniform(bounds[0][0], bounds[0][1], N)
y = random.uniform(bounds[1][0], bounds[1][1], N)

# 区域体积
volume = prod([b - a for a, b in bounds])

# 蒙特卡罗积分
I = volume * mean(f(x, y))

print("二维蒙特卡罗近似积分:", I)
