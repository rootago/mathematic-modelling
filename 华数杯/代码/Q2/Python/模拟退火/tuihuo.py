import numpy as np
import random
import math
from scipy.integrate import solve_ivp


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

def P_sat(T): # Pa 
    return 10**(a - b / ( T + c )) * 1e5 * 1e3

def stop_condition(t,y,T,H,SC,theta):
    return y[0]
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

def calculate_Pp(T, H, SC):

    theta = [5.623560,2.442336e-1,2.214804,1.165577e2,4.245869,1.955567e2,7.284314e3,9.692449e3]

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
def main():
    # 先随机生成一组初始解
    T = random.uniform(27+272.15,153+273.15)
    H = random.uniform(0,1)
    SC = random.uniform(0.06,0.1)

    value = calculate_Pp(T,H,SC)

    tinit = 100
    tmin = 1e-320;
    cooling_rate = 0.95
    t = tinit
    while t > tmin:
         # 生成一组新解
         new_T = T + random.uniform(-20,20)
         new_H = H + random.uniform(-0.1,0.1)
         new_SC = SC + random.uniform(-10,10)

         new_value = calculate_Pp(new_T,new_H,new_SC)

         delta = new_value - value
         if delta < 0 or random.random() <  math.exp(-delta / T):
              T = new_T
              H = new_H
              SC = new_SC
         t = t * cooling_rate
    print("T:")
    print(T)
    print("H:")
    print(H)
    print("SC:")
    print(SC)
    print("value:")
    print(value)
if __name__ == '__main__':
        main()