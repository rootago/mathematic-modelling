import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
import pickle

## 常数 #############################
# Anotine方程参数
a = 6.09451 
b = 2725.96
c = 28.209

# 密度 g / mL
rho_D = 0.944
rho_s = 1.261
rho_c = 1.300
# 粘度 
eta = 0.92e-4 
# 玻尔兹曼常数 g · cm^2  / (K·s^2) 
k = 1.380649e-16
# 液滴半径 cm
r = 1e-4

global m_D0
m_D0 = 24 # g

def P_sat(T): # Pa 
    return 10**(a - b / ( T + c )) * 1e5 * 1e3


def ode_system(t, y ,T ,H , SC ,theta):
    m_D,m_s2,m_form = y
    if(m_D<=1): return [0,0,0]
    k1,kh,Ss,Sc,a1,a2,a3,a4 = theta
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0) 
    
    # 析出计算
    delta_s = max(0,m_s0-m_D*Ss)
    delta_c = max(0,m_c0-m_D*Sc)
    m_s = m_s0 - delta_s
    m_c = m_c0 - delta_c

    # 蒸发
    P = 0 
    V_sol = m_D / rho_D + m_s / rho_s + m_c / rho_c
    dm_dt = - k1 * (P_sat(T) - P)* (m_D / V_sol)*(100-kh*H)/100
    
    # 布朗运动
    phi_c1 = (delta_c / rho_c) / (m_D / rho_D + m_s / rho_s + delta_c / rho_c)
    D = (k*T / (6*np.pi*eta*r)) / (1+2.5*phi_c1)
    v_bar = a1 * np.sqrt(D)

    # 液滴动力学
    dm_form_dt = a2 * v_bar * (delta_s/V_sol) / r

    if(m_form>=2*rho_s*float(4)/float(3)*np.pi*(r**3)):
        global l
        l = 4.3
        dm_s2_dt = a3  * v_bar * (delta_s/V_sol)
    else:
        dm_s2_dt = dm_form_dt
    
    return  [dm_dt,dm_s2_dt,dm_form_dt]

def stop_condition(t,y,T,H,SC,theta):
    return y[0]
stop_condition.terminal = True
stop_condition.direction = -1

def calculate_Pp(T, H, SC, theta):
    y_0 = [m_D0,0,rho_s*float(4)/float(3)*np.pi*(r**3)]
    t_span = [0,1e5]

    sol = solve_ivp(ode_system, t_span, y_0, args=(T,H,SC,theta), 
                    method='RK45', events=stop_condition, rtol=1e-3, atol=1e-6)
    
    m_D_end,m_s2_end,m_form_end = sol.y[:,-1]
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    Pp = theta[7] * m_s2_end  # a4 是比例系数

    # # 新计算方式：大液滴体积占比
    # V_big_droplets = m_s2_end / rho_s
    # V_total_membrane = m_c0 / rho_c + m_s0 / rho_s  # 膜总体积
    # Pp = (V_big_droplets / V_total_membrane) * 100
    return Pp

# 读取数据
df = pd.read_excel("附件2.xlsx")
df = df[11:38]
df = df.drop('Unnamed: 0',axis=1)
df.columns = ['T', 'H', 'SC', 'Pp' ]
df.index = range(len(df))
data =  df.groupby(['T', 'H', 'SC']).mean().reset_index()
data['Pp'] = data['Pp'] / 100
# 存储优化历史的全局变量
optimization_history = {
    'params': [],
    'rmse': [],
    'best_params': None,
    'best_rmse': float('inf')
}

def objective(theta):
    Pp_pred = []
    for _,row in data.iterrows():
        T,H,SC,_ = row
        try:
            Pp = calculate_Pp(T+273.15,H,SC/100,theta)
            Pp_pred.append(Pp)
        except:
            Pp_pred.append(0)  # 处理计算失败的情况
    
    Pp_pred = np.array(Pp_pred)
    rmse = np.sqrt(np.mean((Pp_pred - data['Pp'])**2))
    
    # 记录当前评估
    optimization_history['params'].append(theta.copy())
    optimization_history['rmse'].append(rmse)
    
    # 更新最佳结果
    if rmse < optimization_history['best_rmse']:
        optimization_history['best_rmse'] = rmse
        optimization_history['best_params'] = theta.copy()
        #print(f"New best RMSE: {rmse:.4f}")
    
    return rmse

# 参数边界
bounds = np.array([
    (0, 10),    # k1
    (0, 1),       # kh
    (0,200),     # Ss
    (0,200),     # Sc
    (0, 10),    # a1
    (0, 1000),    # a2
    (0, 10000),    # a3
    (0, 10000)     # a4
])


result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True
)
# 输出最终结果
print("\n===== 优化结果 =====")
print(f"最小 RMSE: {np.sqrt(optimization_history['best_rmse']):.6f}")
print("最优参数:")
param_names = ['k1', 'kh', 'Ss', 'Sc', 'a1', 'a2', 'a3', 'a4']
for name, value in zip(param_names, optimization_history['best_params']):
    print(f"{name}: {value:.6e}")

# 保存优化历史
with open('optimization_history.pkl', 'wb') as f:
    pickle.dump(optimization_history, f)

# 绘制RMSE收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(optimization_history['rmse'])
plt.xlabel('Evaluation Number')
plt.ylabel('RMSE')
plt.title('Optimization Convergence')
plt.yscale('log')
plt.grid(True)
plt.savefig('rmse_convergence.png')
plt.show()

# 计算并保存最终预测结果
final_pred = []
for _, row in data.iterrows():
    T, H, SC, _ = row
    final_pred.append(calculate_Pp(T+273.15, H, SC/100, optimization_history['best_params']))

results_df = pd.DataFrame({
    'T': data['T'],
    'H': data['H'],
    'SC': data['SC'],
    'Actual_Pp': data['Pp'],
    'Predicted_Pp': final_pred
})
results_df.to_csv('final_predictions.csv', index=False)
# 计算R²
ss_res = np.sum((data['Pp'] - final_pred)**2)
ss_tot = np.sum((data['Pp'] - np.mean(data['Pp']))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"模型R²: {r2:.4f}")

plt.figure(figsize=(10, 6))

print(final_pred)
plt.scatter(data['Pp'],final_pred, c='b', alpha=0.6)
plt.show()
