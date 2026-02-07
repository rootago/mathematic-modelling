import numpy as np
from scipy.optimize import dual_annealing

# 1. 定义目标函数（要最小化的函数）
def objective_function(x):
    """
    示例：简单的二次函数
    最优解在 x = [1, 2]，最优值 = 0
    """
    return (x[0] - 1)**2 + (x[1] - 2)**2

# 2. 设置变量的搜索范围
bounds = [(-5, 5),    # 第一个变量的范围
          (-5, 5)]    # 第二个变量的范围

# 3. 运行模拟退火
result = dual_annealing(
    func=objective_function,
    bounds=bounds,
    maxiter=1000,
    seed=42
)

# 4. 显示结果
print(f"最优解: {result.x}")
print(f"最优值: {result.fun}")
print(f"是否成功: {result.success}")
