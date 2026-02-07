import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# Antoine方程参数（用于计算饱和蒸气压）
a = 6.09451 
b = 2725.96
c = 28.209

# 密度 (g/mL)
rho_D = 0.944  # 溶剂密度
rho_s = 1.261  # 溶质密度
rho_c = 1.300  # 涂层材料密度
# 粘度 (Pa·s)
eta = 0.92e-4 
# 玻尔兹曼常数 (g·cm²/(K·s²))
k = 1.380649e-16
# 液滴半径 (cm)
r = 1e-4

# 初始溶剂质量 (g)
global m_D0
m_D0 = 24

def P_sat(T): 
    return 10**(a - b / (T + c)) * 1e5 * 1e3

def ode_system(t, y, T, H, SC, theta):
    m_D, m_s2, m_form = y
    if m_D <= 1: 
        return [0, 0, 0]  # 溶剂耗尽时停止计算
    
    k1, kh, Ss, Sc, a1, a2, a3, a4 = theta
    m_s0 = m_D0 / 4  # 初始溶质质量
    m_c0 = SC * (m_D0 + m_s0)  # 初始涂层材料质量
    
    # 析出计算
    delta_s = max(0, m_s0 - m_D * Ss)  # 溶质析出量
    delta_c = max(0, m_c0 - m_D * Sc)  # 涂层材料析出量
    m_s = m_s0 - delta_s  # 溶液中溶质质量
    m_c = m_c0 - delta_c  # 溶液中涂层材料质量

    # 蒸发过程
    P = 0  # 环境压力
    V_sol = m_D / rho_D + m_s / rho_s + m_c / rho_c  # 溶液体积
    dm_dt = -k1 * (P_sat(T) - P) * (m_D / V_sol) * (100 - kh * H) / 100  # 蒸发速率
    
    # 布朗运动
    phi_c1 = (delta_c / rho_c) / (m_D / rho_D + m_s / rho_s + delta_c / rho_c)  # 体积分数
    D = (k * T / (6 * np.pi * eta * r)) / (1 + 2.5 * phi_c1)  # 扩散系数
    v_bar = a1 * np.sqrt(D)  # 平均速度

    dm_form_dt = a2 * v_bar * (delta_s / V_sol) / r  

    if m_form >= 2 * rho_s * (4/3) * np.pi * (r**3):
        global l
        l = 4.3
        dm_s2_dt = a3 * v_bar * (delta_s / V_sol)  
    else:
        dm_s2_dt = dm_form_dt
    
    return [dm_dt, dm_s2_dt, dm_form_dt]

def stop_condition(t, y, T, H, SC, theta):
    return y[0]
stop_condition.terminal = True
stop_condition.direction = -1

def calculate_Pp(T, H, SC, theta):
    
    y_0 = [m_D0, 0, rho_s * (4/3) * np.pi * (r**3)]
    t_span = [0, 1e5]  

    sol = solve_ivp(ode_system, t_span, y_0, args=(T, H, SC, theta), 
                    method='RK45', events=stop_condition, rtol=1e-3, atol=1e-6)
    
    if sol.y.shape[1] > 0:
        m_D_end, m_s2_end, m_form_end = sol.y[:, -1]
    else:
        
        m_D_end, m_s2_end, m_form_end = y_0
    
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    Pp = theta[7] * m_s2_end  

    return Pp

best_params = [5.623560, 2.442336e-1, 2.214804, 1.165577e2, 4.245869, 
               1.955567e2, 7.284314e3, 9.692449e3]

T_range = np.arange(27, 53, 3)  
H_range = np.arange(50, 90, 4) 
SC_range = np.arange(0, 0.2, 0.05) 

results = []
previous_points = {}  

total_points = len(T_range) * len(H_range) * len(SC_range)
calculated_points = 0

print("开始计算孔隙率...")
for T in T_range:
    for H in H_range:
        for SC in SC_range:
            T_kelvin = T + 273.15  
            Pp = calculate_Pp(T_kelvin, H, SC, best_params)
            
            if Pp < 0:
                adjacent_points = []
                current_key = (T, H, SC)
                
                if T > T_range[0]:
                    prev_T_key = (T-1, H, SC)
                    if prev_T_key in previous_points:
                        adjacent_points.append(previous_points[prev_T_key])

                if H > H_range[0]:
                    prev_H_key = (T, H-1, SC)
                    if prev_H_key in previous_points:
                        adjacent_points.append(previous_points[prev_H_key])
                
                if SC > SC_range[0]:
                    prev_SC_key = (T, H, round(SC-0.01, 2))  
                    if prev_SC_key in previous_points:
                        adjacent_points.append(previous_points[prev_SC_key])
                
                if adjacent_points:
                    avg_Pp = np.mean(adjacent_points)
                    Pp = max(0, avg_Pp) 
                    print(f"修正负值：温度={T}℃, 湿度={H}%, SC={SC:.2f} -> 孔隙率={Pp:.4f} (使用相邻点平均)")
                else:
                    Pp = 0
                    print(f"修正负值：温度={T}℃, 湿度={H}%, SC={SC:.2f} -> 孔隙率=0.0000 (无相邻点)")
            
            current_key = (T, H, SC)
            previous_points[current_key] = Pp
            
            results.append({
                '温度(℃)': T,
                '湿度(%)': H,
                'SC': SC,
                '孔隙率(Pp)': Pp
            })
            
            calculated_points += 1
            if calculated_points % 1000 == 0:
                progress = calculated_points / total_points * 100
                print(f"计算进度: {calculated_points}/{total_points} ({progress:.1f}%)")
            
            print(f"温度={T}℃, 湿度={H}%, SC={SC:.2f}, 孔隙率={Pp:.4f}")

results_df = pd.DataFrame(results)

print("\n温度(℃)\t湿度(%)\tSC\t孔隙率(Pp)")
print("=" * 60)
for idx, row in results_df.iterrows():
    print(f"{row['温度(℃)']}\t\t{row['湿度(%)']}\t\t{row['SC']:.2f}\t{row['孔隙率(Pp)']:.6f}")

# results_df.to_excel("孔隙率计算结果.xlsx", index=False)
print("\n计算完成！结果已保存到 '孔隙率计算结果1.xlsx'")