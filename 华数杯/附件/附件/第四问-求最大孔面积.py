import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing

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
        dm_s2_dt = a3  * v_bar * (delta_s/V_sol)
    else:
        dm_s2_dt = dm_form_dt
    
    return  [dm_dt,dm_s2_dt,dm_form_dt]

def stop_condition(t,y,T,H,SC,theta):
    return y[0]
stop_condition.terminal = True
stop_condition.direction = -1

def calculate_Pp(T, H, SC):

    # theta = [3.261307,2.250802e-1,2.842619e-1,1.527816e2,1.079225,2.165727e2,4.557529e3,2.467350e3]
    # theta = [6.174268,7.375935e-2,2.308716,1.229875e2,8.580233,1.030452e2,6.757047e3,9.063110e3]
    theta = [6.174268385,0.073759348,2.308716099,122.9875061,8.580233011,103.0452159,6757.04748,9063.110353]
    y_0 = [m_D0,0,rho_s*float(4)/float(3)*np.pi*(r**3)]
    t_span = [0,1e5]

    sol = solve_ivp(ode_system, t_span, y_0, args=(T,H,SC,theta), 
                    method='RK45', events=stop_condition, rtol=1e-3, atol=1e-6)
    
    m_D_end,m_s2_end,m_form_end = sol.y[:,-1]
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    Pp = theta[7] * m_s2_end  # a4 是比例系数
    print(Pp)

    # # 新计算方式：大液滴体积占比
    # V_big_droplets = m_s2_end / rho_s
    # V_total_membrane = m_c0 / rho_c + m_s0 / rho_s  # 膜总体积
    # Pp = (V_big_droplets / V_total_membrane) * 100
    return Pp

def objective(x):
    try:
        T, H, SC = x
        T_effect = 987.54
        SC_effect = 630.44
        H_effect = 168.08
        return -calculate_Pp(T, H, SC)
        # return -calculate_Pp(T, H, SC)
    except Exception as e:
        print(f"Exception in objective with x={x}: {e}")
        return 1e10  # 返回一个很差的目标值，避免中断
    
n = 1000

# 变量边界
# bounds = [(30+273.15,50+273.15),  # T边界
#           (50, 90),                   # H边界
#           (0, 0.1)]            # SC边界

bounds = [(30+273.15,50+273.15),  # T边界
          (0.5, 0.9),                   # H边界
          (0.05, 0.11)]            # SC边界

result = dual_annealing(objective, bounds, maxiter=n)


print("Best parameters found:")
print("T =", result.x[0])
print("H =", result.x[1])
print("SC =", result.x[2])
print("Maximum objective function value =", -result.fun)