"""
geneticalgorithm 遗传算法简单模板
"""

import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga

# 1. 定义目标函数
def objective_function(x):
    """
    目标函数 - 需要要最小化的函数
    x: 输入变量数组
    返回: 目标函数值
    """
    # 示例：简单的二次函数
    return np.sum(x**2)

# 2. 设置参数（保持简单）
algorithm_param = {
    'max_num_iteration': 100,     # 最大迭代次数
    'population_size': 50,        # 种群大小
    'mutation_probability': 0.1,  # 变异概率
    'crossover_probability': 0.5, # 交叉概率
}

# 3. 定义变量边界
dimension = 2  # 问题维度
variable_boundaries = np.array([[-10, 10]] * dimension)  # 每个变量的[下界, 上界]

# 4. 创建遗传算法模型
model = ga(
    function=objective_function,
    dimension=dimension,
    variable_type='real',  # 实数类型
    variable_boundaries=variable_boundaries,
    algorithm_parameters=algorithm_param
)

# 5. 运行优化
print("遗传算法开始优化")
print(f"问题维度: {dimension}")
print(f"变量边界: {variable_boundaries.tolist()}")
print(f"种群大小: {algorithm_param['population_size']}")
print(f"最大迭代: {algorithm_param['max_num_iteration']}")

model.run()

# 6. 查看结果
print(f"\n优化完成")
print(f"最优解: {model.output_dict['variable']}")
print(f"最优值: {model.output_dict['function']:.6f}")

# 7. 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(model.report)
plt.title('遗传算法收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('最优适应度值')
plt.grid(True)
plt.show()
