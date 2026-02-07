import numpy as np
import pygad
import matplotlib.pyplot as plt

# 1. 目标函数
def f(x):
    return 10*np.sin(5*x) + 7*np.cos(4*x)

# 2. PyGAD 遗传算法
def fitness_func(solution, solution_idx):
    return f(solution[0])

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=1,
    gene_space={'low':0, 'high':10}
)

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
print("PyGAD 最优 x:", solution[0])
print("PyGAD 最优 f(x):", solution_fitness)

# 获取每代最优适应度
pygad_best_fitness = ga_instance.best_solutions_fitness