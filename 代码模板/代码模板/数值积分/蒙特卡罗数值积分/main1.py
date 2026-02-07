from numpy import random,mean


# 被积函数
# f = lambda x: x**2

# 被积函数展开形式
def f(x):
    return x**2

# 积分区间
a = 0
b = 1

# 随机采样
N = 100000
x = random.uniform(a,b,N)

# 蒙特卡罗积分
I = (b-a)*mean(f(x))

print("蒙特卡罗近似积分:", I)
