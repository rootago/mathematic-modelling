import gurobipy as gp
from gurobipy import Model,Var,GRB
import math

# 参数
customers = [(0,1.5), (2.5,1.2)]
facilities = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
setup_cost = [3,2,3,1,3,3,4,3,2]
cost_per_mile = 1

# 此函数确定设施和客户站点之间的欧式距离

def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return math.sqrt(dx*dx + dy*dy)

# 计算MIP模型的关键参数
num_facilities = len(facilities)
num_customers = len(customers)

# 计算运输成本字典
shipping_cost = {}
for c in range(num_customers):
    for f in range(num_facilities):
        shipping_cost[c,f] = cost_per_mile * compute_distance(customers[c], facilities[f])

# 1.创建模型
model = gp.Model()

# 2.设置决策变量
select = model.addVars(num_facilities,vtype=GRB.BINARY,name='select')
assign = model.addVars(range(num_customers),range(num_facilities),vtype=GRB.CONTINUOUS,lb=0,ub=1)

# 3.更新变量环境
model.update()

# 4.设定目标函数
model.setObjective(select.prod(setup_cost)+assign.prod(shipping_cost),GRB.MINIMIZE)

# 5.设定约束条件
# 约束条件1：每个客户的总分配 = 1
for c in range(num_customers):
    total = 0
    for f in range(num_facilities):
        total += assign[c,f]
    model.addConstr(total == 1, name=f"Demand_{c}")
# 约束条件2：每个客户从仓库分配 <= 仓库是否建
for c in range(num_customers):
    for f in range(num_facilities):
        model.addConstr(assign[(c,f)] <= select[f], name=f"Setup2ship_{c}_{f}")

# 6.执行最优化
model.setParam("MIPGap",0)
model.setParam("Seed",42)
model.setParam('OutputFlag',0)

model.optimize()

# 打印结果
eps = 1e-6
# 总成本
print("最小总成本: ", round(model.ObjVal, 2))

# 仓库建设情况
print("\n=== 仓库建设计划 ===")
for f in range(num_facilities):
    if select[f].x >= eps:   # 视作建立
        print(f"\n 建立仓库的地址为：{f + 1}.")

# 客户运输分配
print("\n=== 客户运输分配 ===")
for c in range(num_customers):
    for f in range(num_facilities):
        if assign[c,f].x >= eps:   # 视作非零分配
            print(f"超市 {c+1} 从工厂 {f+1} 接受 {assign[c,f].x*100:.2f}% 的需求")