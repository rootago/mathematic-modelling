from pyscipopt import Model, quicksum

# 1. 创建模型
model = Model("Template")

# 2. 定义集合
I = [...]    # 集合 I
J = [...]    # 集合 J

# 3. 添加变量（字典形式）
x = {}
for i in I:
    for j in J:
        x[i, j] = model.addVar(vtype="C", name=f"x[{i},{j}]")  # C=连续, I=整数, B=0-1变量

# 4. 添加目标函数
#   Maximize  ∑ c[i,j] * x[i,j]
c = {(i, j): ... for i in I for j in J}  # 系数字典
model.setObjective(
    quicksum(c[i, j] * x[i, j] for i in I for j in J),
    "maximize"   # 或 "minimize"
)

# 5. 添加约束
#   ∑ a[i,j] * x[i,j] ≤ b[i]   for all i
a = {(i, j): ... for i in I for j in J}  # 系数字典
b = {i: ... for i in I}                  # 右端常数
for i in I:
    model.addCons(quicksum(a[i, j] * x[i, j] for j in J) <= b[i], name=f"cons[{i}]")

# 6. 求解
model.optimize()

# 7. 输出结果
status = model.getStatus()
if status == "optimal":
    print("Optimal solution found")
    for i in I:
        for j in J:
            print(f"x[{i},{j}] = {model.getVal(x[i,j])}")
elif status == "infeasible":
    print("Model infeasible")
elif status == "timelimit":
    print("Time limit reached")
else:
    print(f"Optimization ended with status {status}")

# 8. 可选：导出模型
model.writeProblem("model.lp")   # 写成 LP 文件
