import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt

# 常数定义
a = 6.09451  # Antoine方程参数
b = 2725.96
c = 28.209

# 物理性质
rho_D = 0.944   # DMF密度 (g/mL)
rho_s = 1.261   # 溶质密度 (g/mL)
rho_c = 1.300   # 结晶密度 (g/mL)
eta = 0.92e-4   # 粘度 (Pa·s)
k = 1.380649e-16  # 玻尔兹曼常数 (cm^2 g s^{-2} K^{-1})
r = 1e-4        # 液滴半径 (cm)
m_D0 = 24       # 初始溶剂质量 (g)

def P_sat(T):  # 饱和蒸汽压 (Pa)
    """计算溶剂的饱和蒸汽压"""
    return 10**(a - b/(T + c)) * 1e5 * 1e3  # 转换为Pa

def ode_system(t, y, T, H, SC, theta):
    """定义微分方程系统"""
    m_D, m_s2, m_form = y
    
    # 如果溶剂质量过低，停止计算
    if m_D <= 1e-3:
        return [0, 0, 0]
    
    # 解包参数
    k1, kh, Ss, Sc, a1, a2, a3, a4 = theta
    
    # 初始质量计算
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    
    # 计算析出量
    delta_s = max(0, m_s0 - m_D * Ss)
    delta_c = max(0, m_c0 - m_D * Sc)
    m_s = m_s0 - delta_s
    m_c = m_c0 - delta_c
    
    # 计算溶液体积
    V_sol = m_D/rho_D + m_s/rho_s + m_c/rho_c
    
    # 蒸发速率 - 修改了表达式
    P_env = 0  # 环境蒸汽压设为0
    dm_dt = -k1 * (P_sat(T) - P_env) * (m_D / V_sol) * (1 - kh * H / 100)
    
    # 计算扩散系数 - 修改了表达式
    phi_c1 = (delta_c / rho_c) / V_sol
    D = k * T / (6 * np.pi * eta * r * (1 + 2.5 * phi_c1))
    
    # 晶体生长速度
    v_bar = a1 * np.sqrt(D)
    
    # 晶体形成速率 - 修改了表达式
    dm_form_dt = a2 * rho_s * r**2 * v_bar * (delta_s / V_sol)
    
    # 二次成核速率
    single_crystal_mass = rho_s * (4/3) * np.pi * r**3
    if m_form >= 2 * single_crystal_mass:
        dm_s2_dt = a3 * rho_s * r**3 * v_bar * (delta_s / V_sol)
    else:
        dm_s2_dt = dm_form_dt
    
    return [dm_dt, dm_s2_dt, dm_form_dt]

def stop_condition(t, y, T, H, SC, theta):
    """停止条件：溶剂质量过低时停止积分"""
    return y[0] - 1e-3  # 当溶剂质量低于1mg时停止

stop_condition.terminal = True
stop_condition.direction = -1

def calculate_Pp(T, H, SC, theta):
    """计算孔隙压力Pp"""
    # 初始条件 [溶剂质量, 二次成核质量, 晶体形成质量]
    y0 = [m_D0, 0, rho_s * (4/3) * np.pi * r**3]
    
    # 时间范围 (秒)
    t_span = [0, 3600]  # 缩短到1小时
    
    # 求解ODE
    sol = solve_ivp(
        ode_system, t_span, y0,
        args=(T, H, SC, theta),
        method='BDF',  # 改用刚性问题的求解器
        events=stop_condition,
        rtol=1e-6, atol=1e-9
    )
    
    # 获取最终状态
    m_D_end, m_s2_end, m_form_end = sol.y[:, -1]
    
    # 计算孔隙压力 - 使用更合理的表达式
    k1, kh, Ss, Sc, a1, a2, a3, a4 = theta
    Pp = a4 * m_s2_end / (m_D_end + 1e-6)  # 避免除以零
    
    return Pp

# 读取和处理数据
df = pd.read_excel("附件2.xlsx")
df = df.iloc[11:38].drop('Unnamed: 0', axis=1)
df.columns = ['T', 'H', 'SC', 'Pp']
data = df.groupby(['T', 'H', 'SC']).mean().reset_index()

# 目标函数
def objective(theta):
    Pp_pred = []
    for _, row in data.iterrows():
        T, H, SC, _ = row
        # 单位转换：T(K) = T(°C) + 273.15, SC(%→小数)
        Pp_pred.append(calculate_Pp(T + 273.15, H, SC/100, theta))
    
    Pp_pred = np.array(Pp_pred)
    actual = data['Pp'].values
    
    # 使用相对误差作为目标函数，更关注比例而非绝对值
    relative_errors = (Pp_pred - actual) / (actual + 1e-6)
    rmse = np.sqrt(np.mean(relative_errors**2))
    
    print(f"Current RMSE: {rmse:.4f}")
    return rmse

# 参数边界
bounds = [
    (1e-10, 1e-4),   # k1
    (0.01, 1.0),      # kh
    (0.001, 0.1),     # Ss
    (0.001, 0.1),     # Sc
    (1e-6, 1e-2),     # a1
    (1e10, 1e14),     # a2
    (1e10, 1e14),     # a3
    (1e-5, 1e-1)      # a4 (新增比例系数)
]

# 优化算法
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=20,
    mutation=(0.5, 1),
    recombination=0.7,
    tol=0.001,
    disp=True,
    polish=True
)

# 输出结果
optimized_theta = result.x
print(f"Optimized parameters: {optimized_theta}")
print(f"Final RMSE: {result.fun}")

# 使用优化后的参数计算预测值
Pp_pred = []
for _, row in data.iterrows():
    T, H, SC, _ = row
    Pp_pred.append(calculate_Pp(T + 273.15, H, SC/100, optimized_theta))

# 计算R²
actual = data['Pp'].values
ss_res = np.sum((actual - Pp_pred)**2)
ss_tot = np.sum((actual - np.mean(actual))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R²: {r2:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(actual, Pp_pred, c='b', alpha=0.6)
plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
plt.xlabel('Actual Pp')
plt.ylabel('Predicted Pp')
plt.title(f'Pore Pressure Prediction (R² = {r2:.4f})')
plt.grid(True)
plt.show()

# 参数敏感性分析
print("\nParameter Sensitivity:")
param_names = ['k1', 'kh', 'Ss', 'Sc', 'a1', 'a2', 'a3', 'a4']
base_perf = objective(optimized_theta)

for i, name in enumerate(param_names):
    test_theta = optimized_theta.copy()
    test_theta[i] *= 1.1  # 增加10%
    test_perf = objective(test_theta)
    sensitivity = (test_perf - base_perf) / base_perf
    print(f"{name}: {sensitivity:.2%}")