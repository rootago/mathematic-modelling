import numpy as np
import matplotlib.pyplot as plt

# 1. 定义城市坐标
cities = np.array([
    [0, 0],
    [1, 5],
    [5, 2],
    [6, 6],
    [8, 3],
    [2, 7]
])
n_cities = len(cities)

# 计算城市距离矩阵
dist_mat = np.linalg.norm(cities[:, np.newaxis, :] - cities[np.newaxis, :, :], axis=2)

# 2. 遗传算法参数
pop_size = 50
max_gen = 100
mutation_rate = 0.2

# 3. 初始化种群，每个个体是城市排列
population = np.array([np.random.permutation(n_cities) for _ in range(pop_size)])

# 4. 适应度函数（路径长度）
def fitness(route):
    return sum(dist_mat[route[i], route[(i+1) % n_cities]] for i in range(n_cities))

# 5. 主循环
best_cost_history = []
for gen in range(max_gen):
    # 计算适应度
    costs = np.array([fitness(ind) for ind in population])
    
    # 记录最优解
    idx = np.argmin(costs)
    best_route = population[idx]
    best_cost = costs[idx]
    best_cost_history.append(best_cost)
    
    # 选择（轮盘赌）
    fitness_vals = 1 / costs
    prob = fitness_vals / np.sum(fitness_vals)
    cum_prob = np.cumsum(prob)
    
    new_population = []
    while len(new_population) < pop_size:
        # 选择两个父代
        p1 = population[np.searchsorted(cum_prob, np.random.rand())]
        p2 = population[np.searchsorted(cum_prob, np.random.rand())]
        
        # 交叉（OX顺序交叉）
        pt1, pt2 = sorted(np.random.choice(n_cities, 2, replace=False))
        child1 = np.zeros(n_cities, dtype=int)
        child1[pt1:pt2+1] = p1[pt1:pt2+1]
        fill = [city for city in p2 if city not in child1]
        child1[child1==0] = fill
        
        child2 = np.zeros(n_cities, dtype=int)
        child2[pt1:pt2+1] = p2[pt1:pt2+1]
        fill = [city for city in p1 if city not in child2]
        child2[child2==0] = fill
        
        new_population.extend([child1, child2])
    
    # 变异（交换两个城市）
    for i in range(pop_size):
        if np.random.rand() < mutation_rate:
            a, b = np.random.choice(n_cities, 2, replace=False)
            new_population[i][a], new_population[i][b] = new_population[i][b], new_population[i][a]
    
    population = np.array(new_population[:pop_size])

# 6. 输出结果
print("最优路径:", best_route)
print("最短距离:", best_cost)

# 7. 绘制收敛曲线
plt.figure(figsize=(8,5))
plt.plot(best_cost_history, linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('最短距离')
plt.title('遗传算法 TSP 收敛曲线')
plt.grid(True)
plt.show()

# 8. 绘制路径图
plt.figure(figsize=(6,6))
route = np.append(best_route, best_route[0])  # 回到起点
plt.plot(cities[route,0], cities[route,1], '-o', linewidth=2)
plt.title('TSP 最优路径')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
