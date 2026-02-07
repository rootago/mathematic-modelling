from scipy.optimize import differential_evolution, fsolve
from math import pi,atan,sin,cos,sqrt
from numpy import array, cross, linalg, linspace

uavPos = [17800, 0, 1800];
missilePos = [20000, 0, 2000];

alpha1 = atan(missilePos[2]/missilePos[0])
alpha2 = atan(uavPos[2]/uavPos[0])
g = 9.8

O = array([0,207,5])

# 生成圆柱采样点
def generate_cylinder_points():
    # 圆柱参数
    center = [0, 200, 0]  # 底面圆心
    R = 7  # 半径
    H = 10
    
    # 采样精度
    n_r = 5  # 半径方向点数
    n_theta = 36  # 角度方向点数
    n_h = 11  # 高度方向点数
    
    points = []
    
    # 侧面采样
    thetas = linspace(0, 2*pi, n_theta)
    hs = linspace(0, H, n_h)
    
    for theta in thetas:
        for h in hs:
            x = center[0] + R * cos(theta)
            y = center[1] + R * sin(theta)
            z = center[2] + h
            points.append([x, y, z])
    
    # 上底面采样 (z = z0 + H)
    rs = linspace(0, R, n_r)
    for r in rs:
        for theta in thetas:
            x = center[0] + r * cos(theta)
            y = center[1] + r * sin(theta)
            z = center[2] + H
            points.append([x, y, z])
    
    return array(points)

# 生成所有采样点
cylinder_points = generate_cylinder_points()
print(f"生成了 {len(cylinder_points)} 个采样点")

# 1. 定义目标函数
# 传入一个向量 x，返回标量目标值
def objectiveA(x, cylinder_points):
    t1 = x[0]
    t2 = x[1]
    v = x[2]
    theta = x[3]

    results = []

    if t1 >= t2:
        return 1e6

    # 遍历每个采样点 O
    for O in cylinder_points:
        
        def equation1(t):
            """dist1 == 10 的方程"""
            A = array([20000 - (300*t+300*t1+300*t2)*cos(alpha1), 0, 2000 - (300*t+300*t1+300*t2)*sin(alpha1)])
            D = array([17800 + v*(t1+t2)*cos(theta), v*(t1+t2)*sin(theta), 
                        1800 - 0.5*g*t2**2 - 3*t])
            
            OA = A - O
            OD = D - O
            
            if linalg.norm(OA) < 1e-10:
                return 1000
                
            dist1 = linalg.norm(cross(OA, OD)) / linalg.norm(OA)
            # print(dist1-10)
            return dist1 - 10
        # 解方程
        t_solutions_1 = []
        search_range = linspace(0.01, 100, 50)  
        # 求解方程1
        for guess in search_range:
            try:
                # 使用更严格的容差
                t_sol = fsolve(equation1, guess, xtol=1e-10)[0]
                if abs(equation1(t_sol)) < 1e-4 and t_sol > 0:  # 放宽容差
                    t_solutions_1.append(t_sol)
            except Exception as e:
                print(f"Exception for guess {guess}: {type(e).__name__}: {e}")
                continue
        # 去重并排序
        t_solutions_1 = sorted(list(set([round(sol, 6) for sol in t_solutions_1])))
        
        # print(t_solutions_1)

        if len(t_solutions_1) >= 1:
            results.append(t_solutions_1[0])  # 取最小的解
        else:
            results.append(1000)

    # 取出指定的值
    objectiveA = max(results)

    return objectiveA

# 2. 定义变量边界
# 每个变量一个元组 (下界, 上界)
bounds = [(0, 20), (0, 20),  (70, 140), (0, 2*pi)]  

# 3. 调用差分进化求解
    # 用差分进化求解
result = differential_evolution(
    lambda x: objectiveA(x, cylinder_points), 
    bounds,
    strategy='best1bin',
    maxiter=1000,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    disp=False
)

# 4. 输出结果
print("最优解 x =", result.x)
print("最优目标值 f(x) =", result.fun)
