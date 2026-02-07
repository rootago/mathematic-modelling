import gurobipy as gp
from gurobipy import Model,Var,GRB
import math

# 参数
budget = 20
regions, population = gp.multidict({
    0: 523, 1: 690, 2: 420,
    3: 1010, 4: 1200, 5: 850,
    6: 400, 7: 1008, 8: 950
})

sites, coverage, cost = gp.multidict({
    0: [{0,1,5}, 4.2],
    1: [{0,7,8}, 6.1],
    2: [{2,3,4,6}, 5.2],
    3: [{2,5,6}, 5.5],
    4: [{0,2,6,7,8}, 4.8],
    5: [{3,4,8}, 9.2]
})

# 1.创建模型
model = gp.Model()

# 2.设置决策变量
build = model.addVars(len(sites),vtype=GRB.BINARY,name='build')
covered = model.addVars(len(regions),vtype=GRB.BINARY,name='covered')

# 3.更新变量环境
model.update()

# 4.设定目标函数
model.setObjective(covered.prod(population),GRB.MAXIMIZE)

# 5.设定约束条件
# 约束条件1：覆盖：对于每个区域j∈R，确保至少有一个信号塔
for j in range(len(regions)):
    total = 0
    for i in range(len(sites)):
        if j in coverage[i]:
            total += build[i]
    model.addConstr(total >= covered[j], name=f"Build2cover_{j}")
# 约束条件2：预算：我们需要确保建造塔楼的总成本不超过指定的预算
model.addConstr(build.prod(cost)<=budget,name='budget')

# 6.执行最优化
model.setParam("MIPGap",0)
model.setParam("Seed",42)
model.setParam('OutputFlag',0)

model.optimize()

# 打印结果

# 显示决策变量的最佳值
for tower in build.keys():
    if (abs(build[tower].x) > 1e-6):
        print(f"\n 建造基站塔的位置为：{tower}。")

# 建造的信号塔所覆盖的人口百分比
total_population = 0

for region in range(len(regions)):
    total_population += population[region]

coverage = round(100*model.objVal/total_population, 2)
print(f"\n 与基站建设计划相关的人口覆盖率为： {coverage}%")

# 建造基站所需的预算百分比
total_cost = 0

for tower in range(len(sites)):
    if (abs(build[tower].x) > 0.5):
        total_cost += cost[tower]*int(build[tower].x)

budget_consumption = round(100*total_cost/budget, 2)

print(f"\n 与基站建造计划相关的预算消耗百分比为: {budget_consumption} %")