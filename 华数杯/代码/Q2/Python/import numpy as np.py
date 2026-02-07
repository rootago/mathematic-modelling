import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import matplotlib.pyplot as plt

## 常数 #############################
# Anotine方程参数
a = 6.09451 # 无单位
b = 2725.96
c = 28.209

# 溶解度 g / 1g DMF
Ss =  0.01
Sc =  0.01
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
    k1,kh,a1,a2,a3,a4 = theta
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0) # * 100  有问题
    # 当前状态的析出
    delta_s = max(0,m_s0-m_D*Ss)
    delta_c = max(0,m_c0-m_D*Sc)
    m_s = m_s0 - delta_s
    m_c = m_c0 - delta_c

    # 蒸发
    P = 0 # 环境蒸汽压
    V_sol = m_D / rho_D + m_s / rho_s + m_c / rho_c
    dm_dt = - k1 * (P_sat(T) - P)* (m_D / V_sol)*(100-kh*H)/100
    
    
    phi_c1 = (delta_c / rho_c) / (m_D / rho_D + m_s / rho_s + delta_c / rho_c)
    D = (k*T / (6*np.pi*eta*r)) / (1+2.5*phi_c1)

    v_bar = 10**7*a1 * np.sqrt(D)

    dm_form_dt = a2  * (1/r) * v_bar * (delta_s/V_sol) 

    if(m_form>=2*rho_s*float(4)/float(3)*np.pi*(r**3)):
        dm_s2_dt = a3 * v_bar * (delta_s/V_sol)
    else:
        dm_s2_dt = dm_form_dt
    # print(m_D)
    
    return  [dm_dt,dm_s2_dt,dm_form_dt]

end_mass = 1e-3
def stop_condition(t,y,T,H,SC,theta):
    #if(y[0]<=end_mass): print("stop")
    return y[0]

stop_condition.terminal = True
stop_condition.direction = -1

def calculate_Pp(T, H, SC, theta):
    y_0 = [m_D0,0,rho_s*float(4)/float(3)*np.pi*(r**3)]  # m_D(0),

    t_span = [0,1e5]


    sol = solve_ivp(ode_system,
                    
                    t_span,
                    y_0,
                    args = (T,H,SC,theta),
                    method = 'RK45',
                    events=stop_condition,
                    rtol=1e-3, atol=1e-6)
    # solve_ivp的作用：求解方程组
    # print("本次积分已经终止")
    m_D_end,m_s2_end,m_form_end = sol.y[:,-1]
    pass
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)  # * 100
    
    Pp = theta[5] * m_s2_end

    return Pp


df = pd.read_excel("附件2.xlsx")
df = df[11:38]
df = df.drop('Unnamed: 0',axis=1)

df.columns = ['T', 'H', 'SC', 'Pp' ]
df.index = range(len(df))
# print(df)
data =  df.groupby(['T', 'H', 'SC']).mean().reset_index()
pass
def objective(theta):
    Pp_pred = []
    for _,row in data.iterrows():
        T,H,SC,_ = row
        Pp_pred.append(calculate_Pp(T+273.15,H,SC/100,theta))
    print(Pp_pred)
    Pp_pred = np.array(Pp_pred)

    rmse =  np.sqrt(np.mean((Pp_pred - data['Pp'])**2))
    print(rmse)
    return rmse

import scipy.optimize as opt
# theta_0 = [5e13,0.5,5e13,5e13,5e13,1e14]
bounds = np.array([[0,1e4],[0,1],[0,1e4],[0,1e4],[0,1e4],[0,1e4]])

from scipy.optimize import minimize, differential_evolution

theta = differential_evolution(objective,bounds,maxiter=1000, popsize=30, tol=1e-3,disp = True,mutation=0.9,strategy='rand1bin')
print(theta.x)