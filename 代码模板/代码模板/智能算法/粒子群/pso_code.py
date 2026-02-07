import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

# 解决中文字体问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义目标函数（要最小化的函数）
def objective_function(x):
    # 示例：简单的二次函数 f(x,y) = x² + y²
    return np.sum(x**2, axis=1)

# 2. 设置PSO参数
n_particles = 50      # 粒子数量
dimensions = 2        # 问题维度
max_iter = 100        # 最大迭代次数

# PSO超参数
options = {
    'c1': 2.0,    # 个体学习因子
    'c2': 2.0,    # 社会学习因子
    'w': 0.9      # 惯性权重
}

# 搜索边界（可选）
bounds = (np.array([-10, -10]), np.array([10, 10]))  # [下界], [上界]

# 3. 创建优化器并运行
optimizer = ps.single.GlobalBestPSO(
    n_particles=n_particles,
    dimensions=dimensions,
    options=options,
    bounds=bounds
)

# 4. 执行优化
print("开始优化...")
best_cost, best_position = optimizer.optimize(objective_function, iters=max_iter)

# 5. 输出结果
print(f"最优成本: {best_cost}")
print(f"最优位置: {best_position}")

# 6. 绘制收敛曲线
plt.figure(figsize=(8, 5))
plt.plot(optimizer.cost_history)
plt.title('PSO收敛曲线')  # 去掉空格，避免字体问题
plt.xlabel('迭代次数')
plt.ylabel('最优成本')
plt.grid(True)
plt.show()