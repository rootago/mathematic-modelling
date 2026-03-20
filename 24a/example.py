import numpy as np
from scipy.optimize import fsolve
from const import *

def F(x):
    """精确计算 F(x) = x*sqrt(1+x^2) + arcsinh(x)"""
    return x * np.sqrt(1 + x**2) + np.arcsinh(x)

def x_from_t(t, a, b, x0):
    """
    给定时间 t（标量或数组），返回对应的 x（精确数值解）。
    参数：
        t : 要求解的时间点
        a, b : 微分方程常数
        x0 : 初始条件 x(0) = x0
    """
    t = np.asarray(t)
    C = F(x0)                     # 精确常数（t0=0）
    k_target = (2 * b / a) * t + C
    x_sol = np.zeros_like(k_target)

    for i, k in enumerate(k_target.flat):
        # 使用前一个解作为初始猜测（或 x0）
        guess = x0 if i == 0 else x_sol.flat[i-1]
        # 求解方程 F(x) - k = 0
        x_sol.flat[i] = fsolve(lambda x: F(x) - k, guess)[0]
    return x_sol

# 示例：给定 a, b 的具体数值
if __name__ == "__main__":
    a = PITCH / (2 * PI)      # 请根据实际情况修改
    b = VELOCITY      # 请根据实际情况修改
    x0 = 32 * np.pi
    t_vals = np.linspace(0, 10, 5)   # 示例时刻
    x_vals = x_from_t(t_vals, a, b, x0)
    for t, x in zip(t_vals, x_vals):
        print(f"t = {t:.4f}, x = {x:.6f}")