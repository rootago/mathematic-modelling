import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


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
r = 1e-5

global m_D0
m_D0 = 24 # g

T_effect = 987.54
SC_effect = 630.44
H_effect = 168.08

delta = 987.54 - 168.08
T_effect2 = (- 0.05 + (T_effect-H_effect) / delta * 0.1 + 1)  
SC_effect2 = (- 0.05 + (SC_effect-H_effect) / delta * 0.1 + 1)
H_effect2 = (- 0.05 + (H_effect-H_effect) / delta * 0.1 + 1)

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
        dm_s2_dt = a3  * v_bar * (delta_s/V_sol) # * ((m_s2) ** (2/3))
    else:
        dm_s2_dt = dm_form_dt
    
    return  [dm_dt,dm_s2_dt,dm_form_dt]

def stop_condition(t, y, T, H, SC, theta):
    """停止条件：当溶剂耗尽时停止计算"""
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

# 最佳参数（优化得到的结果）
best_params = [6.174268385,	0.073759348,2.308716099,122.9875061,8.580233011,103.0452159,6757.04748,9063.110353]

# 定义遍历范围和步长
T_range = np.arange(27, 53, 1)  # 温度从27℃到52℃，步长1℃
H_range = np.arange(50, 90, 1)  # 湿度从50%到89%，步长1%
SC_range = np.arange(0, 0.2, 0.01)  # 固含量从0.00到0.19，步长0.01

# 创建结果列表
results = []
# 存储已计算点用于平均
previous_points = {}  

# 计算并收集所有组合的孔隙率
total_points = len(T_range) * len(H_range) * len(SC_range)
calculated_points = 0

print("开始计算孔隙率...")
for T in T_range:
    for H in H_range:
        for SC in SC_range:
            T_kelvin = T + 273.15  # 转换为开尔文
            Pp = calculate_Pp(T_kelvin*T_effect2, H*H_effect2/100, SC*SC_effect2, best_params)
            
            # # 如果孔隙率为负，尝试用相邻点平均值修正
            # if Pp < 0:
            #     adjacent_points = []
            #     current_key = (T, H, SC)
                
            #     # 尝试获取相邻点：前一个温度点
            #     if T > T_range[0]:
            #         prev_T_key = (T-1, H, SC)
            #         if prev_T_key in previous_points:
            #             adjacent_points.append(previous_points[prev_T_key])
                
            #     # 尝试获取相邻点：前一个湿度点
            #     if H > H_range[0]:
            #         prev_H_key = (T, H-1, SC)
            #         if prev_H_key in previous_points:
            #             adjacent_points.append(previous_points[prev_H_key])
                
            #     # 尝试获取相邻点：前一个固含量点
            #     if SC > SC_range[0]:
            #         prev_SC_key = (T, H, round(SC-0.01, 2))  # 处理浮点精度
            #         if prev_SC_key in previous_points:
            #             adjacent_points.append(previous_points[prev_SC_key])
                
            #     # 如果有相邻点，计算平均值
            #     if adjacent_points:
            #         avg_Pp = np.mean(adjacent_points)
            #         Pp = max(0, avg_Pp)  # 确保非负
            #         print(f"修正负值：温度={T}℃, 湿度={H}%, SC={SC:.2f} -> 孔隙率={Pp:.4f} (使用相邻点平均)")
            #     else:
            #         # 没有相邻点可用，设为0
            #         Pp = 0
            #         print(f"修正负值：温度={T}℃, 湿度={H}%, SC={SC:.2f} -> 孔隙率=0.0000 (无相邻点)")
            
            # 存储当前点
            current_key = (T, H, SC)
            previous_points[current_key] = Pp
            
            # 添加到结果
            results.append({
                '温度(℃)': T,
                '湿度(%)': H,
                'SC': SC,
                '孔隙率(Pp)': Pp
            })
            
            # 更新进度
            calculated_points += 1
            if calculated_points % 1000 == 0:
                progress = calculated_points / total_points * 100
                print(f"计算进度: {calculated_points}/{total_points} ({progress:.1f}%)")
            
            # 打印当前结果
            print(f"温度={T}℃, 湿度={H}%, SC={SC:.2f}, 孔隙率={Pp:.4f}")

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 输出到终端
print("\n温度(℃)\t湿度(%)\tSC\t孔隙率(Pp)")
print("=" * 60)
for idx, row in results_df.iterrows():
    print(f"{row['温度(℃)']}\t\t{row['湿度(%)']}\t\t{row['SC']:.2f}\t{row['孔隙率(Pp)']:.6f}")

# 保存到Excel文件
results_df.to_excel("孔隙率计算结果2.xlsx", index=False)
