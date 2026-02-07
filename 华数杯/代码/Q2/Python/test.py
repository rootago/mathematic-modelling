import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 物理常数 ==========
rho_d = 0.944    # DMF密度 (g/cm³)
rho_s = 1.26     # 环丁砜密度 (g/cm³)
rho_c = 1.30     # 醋酸纤维素密度 (g/cm³)
eta0 = 0.802e-2  # DMF基础粘度 (g/(cm·s)) @30°C
kB = 1.38e-23    # 玻尔兹曼常数 (J/K)
r = 1e-6         # 小液滴半径 (cm) 假设值

# ========== 全局计数器 ==========
objective_call_count = 0
max_objective_calls = 0  # 稍后赋值

# ========== Antoine方程 ==========
def P_sat(T_C):
    T_K = T_C + 273.15
    return 10**(6.09451 - 2725.96 / (T_K + 28.209))  # bar

# ========== 微分方程组 ==========
def ode_system(t, y, T, H, SC, theta):
    m_D, m_s2 = y
    k1, kh, S_s, S_c, k_form, k_grow = theta
    
    m_s_total = m_D0 / 4
    m_c_total = SC * (m_D0 + m_s_total) * 100
    
    m_s_ppt = max(0, m_s_total - m_D * S_s)
    m_c_ppt = max(0, m_c_total - m_D * S_c)
    m_s1 = m_s_ppt - m_s2
    
    V_total = m_D/rho_d + m_s_total/rho_s + m_c_total/rho_c
    phi_c_ppt = (m_c_ppt / rho_c) / V_total
    eta = eta0 * (1 + 2.5 * phi_c_ppt)
    
    v_vap = k1 * P_sat(T) * (1 - kh * H / 100) * (m_D / V_total)
    
    dm_s2_dt = (k_form * m_s1**2 + k_grow * m_s2 * m_s1) / V_total * np.sqrt(T + 273.15) / np.sqrt(eta)
    
    return [-v_vap, dm_s2_dt]

# ========== 事件函数 ==========
def event_m_D_zero(t, y, T, H, SC, theta):
    return y[0]
event_m_D_zero.terminal = True
event_m_D_zero.direction = -1

# ========== 计算孔面积占比 ==========
def calculate_Pp(T, H, SC, theta):
    global m_D0
    m_D0 = 24
    y0 = [m_D0, 0.0]
    t_span = (0, 10000)
    
    sol = solve_ivp(
        ode_system, t_span, y0,
        args=(T, H, SC, theta),
        events=event_m_D_zero,
        rtol=1e-6, atol=1e-9
    )
    
    m_D_end, m_s2_end = sol.y[:, -1]
    m_s_total = m_D0 / 4
    m_c_total = SC * (m_D0 + m_s_total) * 100
    V_s2 = m_s2_end / rho_s
    V_total_final = m_c_total / rho_c + m_s_total / rho_s
    Pp = (V_s2 / V_total_final) * 100
    return Pp

# ========== 读取数据 ==========
data = pd.read_excel("附件2.xlsx", skiprows=8, usecols="B:E")
data.columns = ['T', 'H', 'SC', 'Pp_exp']
data_new = data[3:]
data_new.index = data_new.index - 3
data_avg = data_new.groupby(['T', 'H', 'SC']).mean().reset_index()

# ========== 定义目标函数（带进度条和轮数打印） ==========
def objective(theta):
    global objective_call_count, max_objective_calls
    objective_call_count += 1
    if max_objective_calls > 0:
        print(f"第 {objective_call_count} 轮 / 共预计 {max_objective_calls} 轮")
    else:
        print(f"第 {objective_call_count} 轮")
    
    Pp_pred = []
    for _, row in tqdm(data_avg.iterrows(), total=len(data_avg), desc="Evaluating RMSE"):
        T, H, SC, Pp_exp = row
        Pp_pred.append(calculate_Pp(T, H, SC / 100, theta))
    Pp_pred = np.array(Pp_pred)
    rmse = np.sqrt(np.mean((Pp_pred - data_avg['Pp_exp'].values)**2))
    return rmse

# ========== 参数优化 ==========
theta0 = [0.1, 0.5, 0.2, 0.1, 0.01, 0.01]
bounds = [(1e-3, 1), (0, 1), (1e-3, 1), (1e-3, 1), (1e-5, 1), (1e-5, 1)]

# 估算最大评估次数（maxiter * popsize）
max_objective_calls = 100 * 15  # 你传给differential_evolution的maxiter和popsize

def callback_func(xk, convergence):
    rmse_now = objective(xk)
    print(f"当前参数: {xk}, 当前RMSE: {rmse_now:.5f}")

result = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-3, callback=callback_func)

theta_opt = result.x
print("优化参数:", theta_opt)
print("RMSE:", result.fun)

# ========== 模型检验（带进度条） ==========
Pp_pred = []
for _, row in tqdm(data_avg.iterrows(), total=len(data_avg), desc="Validating Model"):
    T, H, SC, _ = row
    Pp_pred.append(calculate_Pp(T, H, SC / 100, theta_opt))

plt.figure(figsize=(10, 6))
plt.scatter(data_avg['Pp_exp'], Pp_pred, c='b')
plt.plot([0, 40], [0, 40], 'r--')
plt.xlabel('Experimental Pp (%)')
plt.ylabel('Predicted Pp (%)')
plt.title('Model Validation')
plt.grid(True)
plt.show()

# ========== 最优制备条件 ==========
def objective_func(x):
    T, H, SC = x
    return -calculate_Pp(T, H, SC, theta_opt)

bounds_opt = [(30, 50), (50, 90), (0.06, 0.10)]
result_opt = minimize(objective_func, x0=[40, 70, 0.08],
                      bounds=bounds_opt, method='L-BFGS-B')
T_opt, H_opt, SC_opt = result_opt.x
Pp_opt = -result_opt.fun

print(f"最优条件: T={T_opt:.1f}°C, H={H_opt:.1f}%, SC={SC_opt*100:.1f}%")
print(f"预测孔面积占比: {Pp_opt:.2f}%")

# ========== 影响因素分析 ==========
from statsmodels.formula.api import ols
import statsmodels.api as sm

data_anova = data.copy()
data_anova['T2'] = data_anova['T'] ** 2
data_anova['H2'] = data_anova['H'] ** 2
data_anova['SC2'] = data_anova['SC'] ** 2

model = ols('Pp_exp ~ T + H + SC + T:H + T:SC + H:SC + T2 + H2 + SC2', data=data_anova).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

plt.figure(figsize=(12, 4))
for i, var in enumerate(['T', 'H', 'SC']):
    plt.subplot(1, 3, i + 1)
    group = data.groupby(var)['Pp_exp'].mean()
    plt.plot(group.index, group.values, 'o-')
    plt.xlabel(var)
    plt.ylabel('Mean Pp (%)')
plt.tight_layout()
plt.show()
