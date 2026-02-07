import numpy as np

# 参数
chromosome_length = 10    # 每个染色体基因数
pop_size = 30             # 种群大小
max_gen = 50              # 最大迭代代数
mutation_rate = 0.1       # 变异概率
crossover_rate = 0.8      # 交叉概率

# 每个基因的取值范围 [min, max]
gene_bounds = np.array([[0, 1]] * chromosome_length)

# 初始化种群
pop = np.random.rand(pop_size, chromosome_length)
pop = gene_bounds[:,0] + pop * (gene_bounds[:,1] - gene_bounds[:,0])

# 迭代进化
for gen in range(max_gen):
    # 计算适应度
    fitness = np.array([np.sum(chrom) for chrom in pop]) 

    # 轮盘赌选择
    fitness_shifted = fitness - fitness.min() + 1e-6  # 保证非负
    prob = fitness_shifted / fitness_shifted.sum()
    cum_prob = np.cumsum(prob)

    new_pop = np.zeros_like(pop)
    i = 0
    while i < pop_size:
        # 选择父母
        p1 = pop[np.searchsorted(cum_prob, np.random.rand())]
        p2 = pop[np.searchsorted(cum_prob, np.random.rand())]

        # 交叉
        if np.random.rand() < crossover_rate:
            cp = np.random.randint(1, chromosome_length)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
        else:
            c1, c2 = p1.copy(), p2.copy()

        # 变异
        for chrom in [c1, c2]:
            for j in range(chromosome_length):
                if np.random.rand() < mutation_rate:
                    chrom[j] = gene_bounds[j,0] + (gene_bounds[j,1]-gene_bounds[j,0])*np.random.rand()

        new_pop[i] = c1
        if i+1 < pop_size:
            new_pop[i+1] = c2
        i += 2

    pop = new_pop

    # 输出当前最优
    best_idx = np.argmax(fitness)
    print(f"Generation {gen+1}: Best Fitness = {fitness[best_idx]}")

# 自定义适应度函数示例
def evaluate_fitness(chrom):
    return np.sum(chrom)  
