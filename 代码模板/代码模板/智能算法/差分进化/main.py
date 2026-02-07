from scipy.optimize import differential_evolution

# 1. 定义目标函数
# 传入一个向量 x，返回标量目标值
def objective(x):
    # 示例函数，可替换为你的问题
    # 例如二维函数：f(x, y) = sin(x)*cos(y) + 0.1*(x^2 + y^2)
    return (x[0]-1)**2 + (x[1]+2)**2  # 简单二次函数示例

# 2. 定义变量边界
# 每个变量一个元组 (下界, 上界)
bounds = [(-5, 5), (-5, 5)]  # 对应 x[0] 和 x[1]

# 3. 调用差分进化求解
result = differential_evolution(objective, bounds,
                                strategy='best1bin',   # 变异策略，可选
                                maxiter=1000,          # 最大迭代次数
                                popsize=15,            # 种群规模
                                tol=1e-6,              # 收敛精度
                                mutation=(0.5, 1),     # 变异因子 F
                                recombination=0.7,     # 交叉概率 CR
                                seed=42,               # 随机种子
                                disp=True)             # 显示迭代信息

# 4. 输出结果
print("最优解 x =", result.x)
print("最优目标值 f(x) =", result.fun)
