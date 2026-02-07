import gurobipy as gp
from gurobipy import Model,Var,GRB

# 1.创建模型
model = gp.Model()

# 2.设置决策变量
xA = model.addVar(vtype=GRB.INTEGER,lb=0,name="xA")
xB = model.addVar(vtype=GRB.INTEGER,lb=0,name="xB")
yA = model.addVar(vtype=GRB.BINARY,lb=0,name="yA")
yB = model.addVar(vtype=GRB.BINARY,lb=0,name="yB")

# 3.更新变量环境
model.update()

# 4.设定目标函数
model.setObjective(30*xA+20*xB-60*yA-50*yB,GRB.MAXIMIZE)

# 5.设定约束条件
model.addConstr(20*xA+xB<=100,'con1')
model.addConstr(xA+2*xB<=80,'con2')

model.addConstr(xA<=40*yA,'con3')
model.addConstr(xB<=50*yB,'con4')
model.addConstr(xA>=5*yA,'con5')
model.addConstr(xB>=4*yB,'con6')

# 6.执行最优化
model.setParam("MIPGap",0)
model.setParam("Seed",42)
model.setParam('OutputFlag',0)

model.optimize()

# 打印结果
print("Objective: ",model.ObjVal)
print("xA = ",xA.X,"xB =", xB.X, "yA =", yA.X, "yB =", yB.X)