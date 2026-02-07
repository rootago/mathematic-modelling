from pyscipopt import Model

# 1. 创建模型
model = Model("ProductionPlan")

# 2. 添加变量
x = model.addVar("x", vtype="INTEGER", lb=0.0)
y = model.addVar("y", vtype="INTEGER", lb=0.0)

# 3. 设置目标函数（最大化）
model.setObjective(40 * x + 30 * y, "maximize")

# 4. 添加约束
model.addCons(2 * x + y <= 8, name="time")
model.addCons(3 * x + 2 * y <= 10, name="material")

# 5. 参数设置
model.setParam("randomization/permutationseed", 42)  # 随机种子

# 6. 求解
model.optimize()

# 7. 输出结果
status = model.getStatus()

# 常见状态说明：
# optimal     最优解
# infeasible  无解
# unbounded   无界
# timelimit   时间上限
# interrupted 被中断
# unknown     未知（可能是提前停止）

if status == "optimal":
    print("Optimal Objective Value:", model.getObjVal())
    print(f"x = {model.getVal(x):.0f}")
    print(f"y = {model.getVal(y):.0f}")
elif status == "infeasible":
    print("Model infeasible")
elif status == "unbounded":
    print("Model unbounded")
elif status == "timelimit":
    print(f"Time limit reached. Best Obj = {model.getObjVal()}")
else:
    print(f"Optimization ended with status {status}")

# 8. 导出模型
model.writeProblem("model.lp")   # 导出 LP 格式
