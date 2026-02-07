import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    theta = [2.361832,0.100319,2.298162,101.4566,5.170484,64.3464,3153.973,9690.302]
    y_0 = [m_D0,0,rho_s*float(4)/float(3)*np.pi*(r**3)]
    t_span = [0,1e5]

    sol = solve_ivp(ode_system, t_span, y_0, args=(T,H,SC,theta), 
                    method='RK45', events=stop_condition, rtol=1e-3, atol=1e-6)
    
    m_D_end,m_s2_end,m_form_end = sol.y[:,-1]
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    Pp = theta[7] * m_s2_end  # a4 是比例系数
    print(Pp)
    return Pp

# 修改目标函数以记录优化过程
optimization_history = []

def objective_with_history(x):
    try:
        T, H, SC = x
        Pp = calculate_Pp(T, H, SC)
        optimization_history.append({
            'T': T,
            'H': H, 
            'SC': SC,
            'Pp': Pp,
            'iteration': len(optimization_history)
        })
        return -Pp
    except Exception as e:
        print(f"Exception in objective with x={x}: {e}")
        return 1e10

def run_optimization_with_visualization():
    global optimization_history
    optimization_history = []  # 重置历史记录
    
    # 变量边界
    bounds = [(30+273.15, 50+273.15),  # T边界
              (50, 90),                 # H边界
              (0.06, 0.1)]                # SC边界
    
    n = 3000  # 迭代次数
    
    print("开始优化")
    result = dual_annealing(objective_with_history, bounds, maxiter=n)
    
    print("Best parameters found:")
    print("T =", result.x[0])
    print("H =", result.x[1]) 
    print("SC =", result.x[2])
    print("Maximum Pp value =", -result.fun)
    
    return result

def plot_convergence_pair():
    if len(optimization_history) == 0:
        print("没有优化历史数据，请先运行优化")
        return

    df = pd.DataFrame(optimization_history)

    cumulative_best = []
    best_so_far = -np.inf
    for pp in df['Pp']:
        if pp > best_so_far:
            best_so_far = pp
        cumulative_best.append(best_so_far)

    convergence_threshold = 1e-6
    window_size = 100
    convergence_point = len(df)

    if len(cumulative_best) > window_size * 2:
        for i in range(window_size, len(cumulative_best) - window_size):
            current_best = cumulative_best[i]
            future_best = cumulative_best[i + window_size]
            relative_improvement = (future_best - current_best) / max(abs(current_best), 1e-10)
            if relative_improvement < convergence_threshold:
                convergence_point = i + window_size * 2
                break

    # 截取到收敛点的数据
    plot_df = df.iloc[:convergence_point].copy()
    plot_cumulative = cumulative_best[:convergence_point]

    # 开始绘图：上下两个子图 
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # 子图1
    axes[0].plot(plot_df['iteration'], plot_df['Pp'], 'lightblue', alpha=0.6, label='所有评估点', linewidth=1)
    axes[0].plot(plot_df['iteration'], plot_cumulative, 'red', linewidth=3, label='最佳值收敛')

    # 标记收敛点
    if convergence_point < len(df):
        axes[0].axvline(x=convergence_point, color='orange', linestyle='--', alpha=0.8, linewidth=2)
        axes[0].text(convergence_point + len(plot_df)*0.05, max(plot_cumulative)*0.9,
                     f'收敛点\n(第{convergence_point}次迭代)',
                     fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))

    # 标注全局最优
    final_best_all = df['Pp'].max()
    final_best_shown = max(plot_cumulative) if len(plot_cumulative) > 0 else final_best_all
    axes[0].axhline(y=final_best_all, color='red', linestyle=':', alpha=0.7)
    axes[0].text(len(plot_df)*0.7 if len(plot_df) > 0 else 0, final_best_all*1.05,
                 f'全局最佳Pp值: {final_best_all:.6f}',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    if abs(final_best_shown - final_best_all) > 1e-6:
        axes[0].text(len(plot_df)*0.02 if len(plot_df) > 0 else 0, final_best_all*0.8,
                     f'注意：图中显示到第{convergence_point}次评估\n实际最优值在后续迭代中找到',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    # 收敛统计信息
    total_evaluations = len(df)
    efficiency = (convergence_point / total_evaluations) * 100 if total_evaluations > 0 else 0
    axes[0].text(len(plot_df)*0.02 if len(plot_df) > 0 else 0, max(plot_cumulative) * 0.7 if len(plot_cumulative) > 0 else final_best_all*0.7,
                 f'收敛效率: {efficiency:.1f}%\n({convergence_point}/{total_evaluations}次评估)',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    axes[0].set_ylabel('Pp值', fontsize=14)
    axes[0].set_title(f'优化收敛过程 (显示前{convergence_point}次评估)', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 子图2 
    axes[1].plot(df['iteration'], df['Pp'], 'lightblue', alpha=0.6, linewidth=1, label='所有评估点')
    axes[1].plot(df['iteration'], cumulative_best, 'red', linewidth=3, label='最佳值收敛')
    axes[1].axhline(y=final_best_all, color='red', linestyle=':', alpha=0.7)
    axes[1].text(len(df)*0.7 if len(df) > 0 else 0, final_best_all*1.05,
                 f'全局最佳Pp值: {final_best_all:.6f}',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    axes[1].set_xlabel('函数评估次数', fontsize=14)
    axes[1].set_ylabel('Pp值', fontsize=14)
    axes[1].set_title('优化收敛过程（完整版本）', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_parameter_space():
    """图2: 3D参数空间探索图"""
    if len(optimization_history) == 0:
        print("没有优化历史数据，请先运行优化")
        return
    
    df = pd.DataFrame(optimization_history)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据Pp值着色
    scatter = ax.scatter(df['T'], df['H'], df['SC'], 
                        c=df['Pp'], cmap='viridis', alpha=0.7, s=50)
    
    # 标记最优点
    best_idx = df['Pp'].idxmax()
    ax.scatter(df.loc[best_idx, 'T'], df.loc[best_idx, 'H'], df.loc[best_idx, 'SC'], 
               c='red', s=200, marker='*', label='最优解')
    
    ax.set_xlabel('温度 T (K)', fontsize=12)
    ax.set_ylabel('湿度 H (%)', fontsize=12)
    ax.set_zlabel('浓度 SC', fontsize=12)
    ax.set_title('参数空间探索轨迹', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, shrink=0.8, pad=0.1)
    cbar.set_label('Pp值', fontsize=12)
    
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_parameter_evolution():
    """图3: 参数演化图"""
    if len(optimization_history) == 0:
        print("没有优化历史数据，请先运行优化")
        return
    
    df = pd.DataFrame(optimization_history)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 温度演化
    axes[0].plot(df['iteration'], df['T'], 'red', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('温度 T (K)', fontsize=12)
    axes[0].set_title('参数演化过程', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 湿度演化
    axes[1].plot(df['iteration'], df['H'], 'green', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('湿度 H (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 浓度演化
    axes[2].plot(df['iteration'], df['SC'], 'blue', linewidth=2, alpha=0.8)
    axes[2].set_ylabel('浓度 SC', fontsize=12)
    axes[2].set_xlabel('迭代次数', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # 标记最优点
    best_idx = df['Pp'].idxmax()
    for i, param in enumerate(['T', 'H', 'SC']):
        axes[i].axvline(x=best_idx, color='red', linestyle='--', alpha=0.7)
        axes[i].scatter(best_idx, df.loc[best_idx, param], color='red', s=100, zorder=5)
    
    plt.tight_layout()
    plt.show()

def show_optimization_summary():
    """显示优化结果摘要"""
    if len(optimization_history) == 0:
        print("没有优化历史数据")
        return
    
    df = pd.DataFrame(optimization_history)
    
    print("\n" + "="*50)
    print("           优化结果摘要")
    print("="*50)
    print(f"总迭代次数: {len(df)}")
    print(f"最优Pp值: {df['Pp'].max():.6f}")
    
    best_idx = df['Pp'].idxmax()
    print(f"\n最优参数:")
    print(f"  温度 T: {df.loc[best_idx, 'T']:.2f} K ({df.loc[best_idx, 'T']-273.15:.2f}°C)")
    print(f"  湿度 H: {df.loc[best_idx, 'H']:.2f} %")
    print(f"  浓度 SC: {df.loc[best_idx, 'SC']:.6f}")
    print(f"  在第 {best_idx+1} 次迭代找到")
    
    # 计算改进程度
    initial_best = df['Pp'].iloc[:50].max()  # 前50次的最佳值
    final_best = df['Pp'].max()
    improvement = ((final_best - initial_best) / initial_best) * 100
    print(f"\n性能改进: {improvement:.2f}%")
    print("="*50)

def main():
    """主函数 - 运行优化并依次显示关键图表"""
    
    print("开始运行优化算法")
    
    try:
        result = run_optimization_with_visualization()
        
        if len(optimization_history) < 10:
            return result
        
        # 显示结果摘要
        show_optimization_summary()

        # 依次显示关键图表
        print("\n1. 收敛过程图")
        plot_convergence_pair()
        
        input("按Enter键继续查看下一个图表")
        
        print("\n2. 3D参数空间探索图")
        plot_parameter_space()
        
        input("按Enter键继续查看下一个图表")
        
        print("\n3. 参数演化图")
        plot_parameter_evolution()
        
        input("按Enter键继续查看下一个图表")
        
        print("\n所有可视化完成！")
        
        return result
    
    except KeyboardInterrupt:
        print("\n计算中断")
        if len(optimization_history) > 0:
            show_optimization_summary()
            plot_convergence_pair()
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 运行主函数
if __name__ == "__main__":
    result = main()