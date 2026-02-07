import gurobipy as gp
from gurobipy import Model, GRB

# 1. 创建模型
model: Model = gp.Model("ProductionPlan")

# 2. 添加变量
x = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="x")
y = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="y")
model.update()

# 3. 设置目标函数
model.setObjective(40 * x + 30 * y, GRB.MAXIMIZE)

# 4. 添加约束
model.addConstr(2 * x + y <= 8, name="time")
model.addConstr(3 * x + 2 * y <= 10, name="material")

# 5. 参数设置（可选）
model.setParam("OutputFlag", 1)   # 是否打印日志
model.setParam("TimeLimit", 30)   # 限时 30 秒
model.setParam("MIPGap", 0.0)     # 求精确解
model.setParam("Seed", 42)        # 随机种子

# 6. 求解
model.optimize()

# 7. 输出结果
status = model.Status

# 常见状态说明：
# GRB.OPTIMAL     → 最优解
# GRB.INFEASIBLE  → 无解，约束冲突（用 computeIIS() 查原因）
# GRB.UNBOUNDED   → 无界，目标函数无限大/小
# GRB.TIME_LIMIT  → 到达时限，有当前最好解和界
# GRB.INTERRUPTED → 被中断
# GRB.SUBOPTIMAL  → 次优解

if status == GRB.OPTIMAL:
    print("Optimal Objective Value:", model.ObjVal)
    print(f"x = {x.X:.0f}")
    print(f"y = {y.X:.0f}")
elif status == GRB.INFEASIBLE:
    print("Model infeasible, computing IIS...")
    model.computeIIS()
    model.write("infeasible.ilp")
elif status == GRB.TIME_LIMIT:
    print(f"Time limit reached. Best Obj = {model.ObjVal}, Bound = {model.ObjBound}")
else:
    print(f"Optimization ended with status {status}")

# 8. 导出模型与解
model.write("model.lp")
model.write("model.sol")
