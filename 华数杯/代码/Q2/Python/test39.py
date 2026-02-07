
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建目录保存结果
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/visualizations'):
    os.makedirs('results/visualizations')

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

T_effect = 987.54
SC_effect = 630.44
H_effect = 168.08

delta = 987.54 - 168.08
T_effect2 = (- 0.05 + (T_effect-H_effect) / delta * 0.1 + 1)  
SC_effect2 = (- 0.05 + (SC_effect-H_effect) / delta * 0.1 + 1)
H_effect2 = (- 0.05 + (H_effect-H_effect) / delta * 0.1 + 1)

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
        dm_s2_dt = a3  * v_bar * (delta_s/V_sol)* ((m_s2) ** (2/3))
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

    return Pp

# 读取数据
df = pd.read_excel("附件2.xlsx")
df = df[11:38]
df = df.drop('Unnamed: 0',axis=1)
df.columns = ['T', 'H', 'SC', 'Pp' ]
df.index = range(len(df))
data =  df.groupby(['T', 'H', 'SC']).mean().reset_index()
data['Pp'] = data['Pp'] / 100

# # 可视化原始数据
# plt.figure(figsize=(15, 10))
# plt.suptitle('原始数据分布和关系', fontsize=16)

# # 温度分布
# plt.subplot(2, 2, 1)
# sns.histplot(data['T'], kde=True, color='skyblue')
# plt.title('温度分布')
# plt.xlabel('温度(摄氏度)')

# # 湿度分布
# plt.subplot(2, 2, 2)
# sns.histplot(data['H'], kde=True, color='salmon')
# plt.title('湿度分布')
# plt.xlabel('湿度 (%)')

# # SC分布
# plt.subplot(2, 2, 3)
# sns.histplot(data['SC'], kde=True, color='lightgreen')
# plt.title('SC分布')
# plt.xlabel('SC')

# # Pp分布
# plt.subplot(2, 2, 4)
# sns.histplot(data['Pp'], kde=True, color='gold')
# plt.title('Pp分布')
# plt.xlabel('Pp')

# plt.tight_layout()
# plt.savefig('results/visualizations/data_distribution.png')
# plt.close()

# 变量关系图
# plt.figure(figsize=(15, 10))
# plt.suptitle('变量间关系', fontsize=16)

# # T vs Pp
# plt.subplot(2, 2, 1)
# sns.scatterplot(data=data, x='T', y='Pp', hue='H', palette='viridis', size='SC')
# plt.title('温度 vs Pp')

# # H vs Pp
# plt.subplot(2, 2, 2)
# sns.scatterplot(data=data, x='H', y='Pp', hue='T', palette='cool', size='SC')
# plt.title('湿度 vs Pp')

# # SC vs Pp
# plt.subplot(2, 2, 3)
# sns.scatterplot(data=data, x='SC', y='Pp', hue='T', palette='plasma', size='H')
# plt.title('SC vs Pp')

# # 3D图
# ax = plt.subplot(2, 2, 4, projection='3d')
# sc = ax.scatter(data['T'], data['H'], data['Pp'], c=data['SC'], cmap='viridis', s=data['SC']*5)
# ax.set_xlabel('温度 (°C)')
# ax.set_ylabel('湿度 (%)')
# ax.set_zlabel('Pp')
# plt.title('T-H-Pp关系 (颜色表示SC)')
# plt.colorbar(sc, label='SC')

# plt.tight_layout()
# plt.savefig('results/visualizations/variable_relationships.png')
# plt.close()

# 存储优化历史的全局变量
optimization_histories = []

# 参数边界
bounds = np.array([
    (0, 10),    # k1
    (0, 1),     # kh
    (0, 200),   # Ss
    (0, 200),   # Sc
    (0, 10),    # a1
    (0, 1000),  # a2
    (0, 10000), # a3
    (0, 10000)  # a4
])

# 运行10次优化
for run in range(10):
    print(f"\n===== 开始优化运行 #{run+1} =====")
    
    # 当前运行的优化历史
    optimization_history = {
        'params': [],
        'rmse': [],
        'best_params': None,
        'best_rmse': float('inf'),
        'run': run
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
            print(f"运行 #{run+1} - 新最佳 RMSE: {rmse:.6f}")
        
        return rmse
    
    # 运行差分进化算法
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )
    
    # 保存本次运行的优化历史
    optimization_histories.append(optimization_history)
    
    # 保存优化历史到文件
    with open(f'results/optimization_run_{run+1}.pkl', 'wb') as f:
        pickle.dump(optimization_history, f)
    
    # 绘制本次运行的收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_history['rmse'])
    plt.xlabel('评估次数')
    plt.ylabel('RMSE')
    plt.title(f'优化运行 #{run+1} - 收敛曲线')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'results/visualizations/convergence_run_{run+1}.png')
    plt.close()
    
    # 计算并保存最终预测结果
    final_pred = []
    for _, row in data.iterrows():
        T, H, SC, _ = row
        final_pred.append(calculate_Pp(T+273.15, H, SC/100, optimization_history['best_params']))
    
    ss_res = np.sum((data['Pp'] - final_pred)**2)
    ss_tot = np.sum((data['Pp'] - np.mean(data['Pp']))**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"运行 #{run+1} - 模型R2: {r2:.4f}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'T': data['T'],
        'H': data['H'],
        'SC': data['SC'],
        'Actual_Pp': data['Pp'],
        'Predicted_Pp': final_pred,
        'Error': data['Pp'] - np.array(final_pred)
    })
    results_df.to_csv(f'results/predictions_run_{run+1}.csv', index=False)
    
    # 绘制预测与实际值对比
    plt.figure(figsize=(10, 8))
    plt.scatter(data['Pp'], final_pred, c='blue', alpha=0.6)
    plt.plot([0, max(data['Pp'])], [0, max(data['Pp'])], 'r--')
    plt.xlabel('实际Pp')
    plt.ylabel('预测Pp')
    plt.title(f'运行 #{run+1} - 预测值与实际值对比 (R2={r2:.4f})')
    plt.grid(True)
    
    # 添加误差分布小图
    ax_inset = plt.axes([0.6, 0.2, 0.3, 0.3])
    sns.histplot(results_df['Error'], kde=True, ax=ax_inset, color='green')
    ax_inset.set_title('误差分布')
    ax_inset.set_xlabel('误差')
    
    plt.tight_layout()
    plt.savefig(f'results/visualizations/prediction_vs_actual_run_{run+1}.png')
    plt.close()

# 汇总所有运行结果
summary = {
    'Run': [],
    'RMSE': [],
    'R2': [],
    'k1': [],
    'kh': [],
    'Ss': [],
    'Sc': [],
    'a1': [],
    'a2': [],
    'a3': [],
    'a4': []
}

for i, history in enumerate(optimization_histories):
    summary['Run'].append(i+1)
    summary['RMSE'].append(history['best_rmse'])
    
    # 计算本次运行的R2
    final_pred = []
    for _, row in data.iterrows():
        T, H, SC, _ = row
        final_pred.append(calculate_Pp((T+273.15)*T_effect2, H*H_effect2, SC/100*SC_effect2, history['best_params']))
    
    ss_res = np.sum((data['Pp'] - final_pred)**2)
    ss_tot = np.sum((data['Pp'] - np.mean(data['Pp']))**2)
    r2 = 1 - (ss_res / ss_tot)
    summary['R2'].append(r2)
    
    # 记录参数
    params = history['best_params']
    summary['k1'].append(params[0])
    summary['kh'].append(params[1])
    summary['Ss'].append(params[2])
    summary['Sc'].append(params[3])
    summary['a1'].append(params[4])
    summary['a2'].append(params[5])
    summary['a3'].append(params[6])
    summary['a4'].append(params[7])

# 保存汇总结果
summary_df = pd.DataFrame(summary)
summary_df.to_csv('results/optimization_summary.csv', index=False)

# 绘制所有运行的RMSE和R2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=summary_df, x='Run', y='RMSE', palette='viridis')
plt.title('各次优化的RMSE')
plt.yscale('log')

plt.subplot(1, 2, 2)
sns.barplot(data=summary_df, x='Run', y='R2', palette='coolwarm')
plt.title('各次优化的R2')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('results/visualizations/all_runs_metrics.png')
plt.close()

# 绘制参数分布
param_names = ['k1', 'kh', 'Ss', 'Sc', 'a1', 'a2', 'a3', 'a4']
plt.figure(figsize=(16, 12))
for i, param in enumerate(param_names):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=summary_df[param])
    plt.title(f'{param} 分布')
    plt.ylabel(param)

plt.tight_layout()
plt.savefig('results/visualizations/parameter_distributions.png')
plt.close()

# 绘制参数相关性热力图
plt.figure(figsize=(12, 10))
corr = summary_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('参数与性能指标相关性')
plt.savefig('results/visualizations/parameter_correlation.png')
plt.close()

# 找出最佳运行
best_run_idx = np.argmin(summary_df['RMSE'])
best_params = optimization_histories[best_run_idx]['best_params']
print(f"\n===== 全局最佳结果 (运行 #{best_run_idx+1}) =====")
print(f"最小 RMSE: {summary_df.loc[best_run_idx, 'RMSE']:.6f}")
print(f"R2: {summary_df.loc[best_run_idx, 'R2']:.4f}")
print("最优参数:")
for name, value in zip(param_names, best_params):
    print(f"{name}: {value:.6e}")

# 保存最佳参数
best_params_df = pd.DataFrame([best_params], columns=param_names)
best_params_df.to_csv('results/best_parameters.csv', index=False)

print("\n===== 优化完成 =====")
print(f"所有结果已保存到 'results' 目录")