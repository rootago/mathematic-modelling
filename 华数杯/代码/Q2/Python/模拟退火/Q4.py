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

## 常数 #############################
# Antoine方程参数
a = 6.09451
b = 2725.96
c = 28.209

# 密度 g / mL
rho_D = 0.944
rho_s = 1.261
rho_c = 1.300

# 粘度
eta = 0.92e-4

# 玻尔兹曼常数 g · cm^2 / (K·s^2)
k = 1.380649e-16

# 液滴半径 cm
r = 1e-4

m_D0 = 24  # g

def P_sat(T):  # Pa
    return 10**(a - b / (T + c)) * 1e5 * 1e3

def stop_condition(t, y, T, H, SC, theta):
    return y[0]

def ode_system(t, y, T, H, SC, theta):
    m_D, m_s2, m_form = y
    if(m_D <= 1): 
        return [0, 0, 0]
    
    k1, kh, Ss, Sc, a1, a2, a3, a4 = theta
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    
    # 析出计算
    delta_s = max(0, m_s0 - m_D * Ss)
    delta_c = max(0, m_c0 - m_D * Sc)
    m_s = m_s0 - delta_s
    m_c = m_c0 - delta_c
    
    # 蒸发
    P = 0
    V_sol = m_D / rho_D + m_s / rho_s + m_c / rho_c
    dm_dt = -k1 * (P_sat(T) - P) * (m_D / V_sol) * (100 - kh * H) / 100
    
    # 布朗运动
    phi_c1 = (delta_c / rho_c) / (m_D / rho_D + m_s / rho_s + delta_c / rho_c)
    D = (k * T / (6 * np.pi * eta * r)) / (1 + 2.5 * phi_c1)
    v_bar = a1 * np.sqrt(D)
    
    # 液滴动力学
    dm_form_dt = a2 * v_bar * (delta_s / V_sol) / r
    
    if(m_form >= 2 * rho_s * float(4) / float(3) * np.pi * (r**3)):
        global l
        l = 4.3
        dm_s2_dt = a3 * v_bar * (delta_s / V_sol)
    else:
        dm_s2_dt = dm_form_dt
    
    return [dm_dt, dm_s2_dt, dm_form_dt]

def calculate_Pp(T, H, SC):
    theta = [5.623560, 2.442336e-1, 2.214804, 1.165577e2, 4.245869, 1.955567e2, 7.284314e3, 9.692449e3]
    y_0 = [m_D0, 0, rho_s * float(4) / float(3) * np.pi * (r**3)]
    t_span = [0, 1e5]
    
    sol = solve_ivp(ode_system, t_span, y_0, args=(T, H, SC, theta),
                    method='RK45', events=stop_condition, rtol=1e-3, atol=1e-6)
    
    m_D_end, m_s2_end, m_form_end = sol.y[:, -1]
    m_s0 = m_D0 / 4
    m_c0 = SC * (m_D0 + m_s0)
    Pp = theta[7] * m_s2_end  # a4 是比例系数
    
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
              (0, 0.1)]                # SC边界
    
    n = 3000  # 迭代次数
    
    print("开始优化...")
    result = dual_annealing(objective_with_history, bounds, maxiter=n)
    
    print("Best parameters found:")
    print("T =", result.x[0])
    print("H =", result.x[1]) 
    print("SC =", result.x[2])
    print("Maximum Pp value =", -result.fun)
    
    return result

def plot_convergence():
    """图1: 收敛过程图 - 只显示到收敛稳定为止"""
    if len(optimization_history) == 0:
        print("没有优化历史数据，请先运行优化")
        return
    
    df = pd.DataFrame(optimization_history)
    
    # 计算累积最优值
    cumulative_best = []
    best_so_far = -np.inf
    for pp in df['Pp']:
        if pp > best_so_far:
            best_so_far = pp
        cumulative_best.append(best_so_far)
    
    # 找到收敛点：连续100次迭代没有明显改进就认为收敛
    convergence_threshold = 1e-6  # 改进阈值
    window_size = 100  # 观察窗口
    convergence_point = len(df)  # 默认到最后
    
    if len(cumulative_best) > window_size * 2:
        for i in range(window_size, len(cumulative_best) - window_size):
            # 检查后续window_size个点的改进
            current_best = cumulative_best[i]
            future_best = cumulative_best[i + window_size]
            relative_improvement = (future_best - current_best) / max(abs(current_best), 1e-10)
            
            if relative_improvement < convergence_threshold:
                convergence_point = i + window_size * 2  # 再多画一点，确保看到稳定
                break
    
    # 截取数据到收敛点
    plot_df = df.iloc[:convergence_point].copy()
    plot_cumulative = cumulative_best[:convergence_point]
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_df['iteration'], plot_df['Pp'], 'lightblue', alpha=0.6, label='所有评估点', linewidth=1)
    plt.plot(plot_df['iteration'], plot_cumulative, 'red', linewidth=3, label='最佳值收敛')
    
    # 标记收敛点
    if convergence_point < len(df):
        plt.axvline(x=convergence_point, color='orange', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(convergence_point + len(plot_df)*0.05, max(plot_cumulative)*0.9, 
                 f'收敛点\n(第{convergence_point}次迭代)', 
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    plt.xlabel('函数评估次数', fontsize=14)
    plt.ylabel('Pp值', fontsize=14)
    plt.title(f'优化收敛过程 (显示前{convergence_point}次评估)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加最终结果标注（使用全部数据的最优值）
    final_best_all = df['Pp'].max()  # 使用全部数据的最优值
    final_best_shown = max(plot_cumulative)  # 图中显示的最优值
    
    plt.axhline(y=final_best_all, color='red', linestyle=':', alpha=0.7)
    plt.text(len(plot_df)*0.7, final_best_all*1.05, f'全局最佳Pp值: {final_best_all:.6f}', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 如果截取部分的最优值和全局最优值不同，添加说明
    if abs(final_best_shown - final_best_all) > 1e-6:
        plt.text(len(plot_df)*0.02, final_best_all*0.8, 
                 f'注意：图中显示到第{convergence_point}次评估\n实际最优值在后续迭代中找到', 
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # 添加收敛统计信息
    total_evaluations = len(df)
    efficiency = (convergence_point / total_evaluations) * 100
    plt.text(len(plot_df)*0.02, max(plot_cumulative)*0.7, 
             f'收敛效率: {efficiency:.1f}%\n({convergence_point}/{total_evaluations}次评估)', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print(f"收敛分析：算法在第{convergence_point}次评估时基本稳定")
    print(f"收敛效率：{efficiency:.1f}% 的计算量达到当时的最优解")
    if abs(df['Pp'].max() - max(plot_cumulative)) > 1e-6:
        print(f"注意：真正的全局最优解出现在第{df['Pp'].idxmax()+1}次评估")
        print(f"全局最优值：{df['Pp'].max():.6f}，图中最后显示值：{max(plot_cumulative):.6f}")
    
    return convergence_point

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

def plot_contour_map():
    """图4: 等高线图 - 显示目标函数地形"""
    print("正在计算等高线图数据，请稍等...")
    
    # 创建网格
    T_range = np.linspace(30+273.15, 50+273.15, 25)
    H_range = np.linspace(50, 90, 25)
    SC_fixed = 0.05  # 固定SC值
    
    T_grid, H_grid = np.meshgrid(T_range, H_range)
    Pp_grid = np.zeros_like(T_grid)
    
    # 计算网格上的Pp值
    total_points = T_grid.shape[0] * T_grid.shape[1]
    calculated = 0
    
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            try:
                Pp_grid[i, j] = calculate_Pp(T_grid[i, j], H_grid[i, j], SC_fixed)
            except:
                Pp_grid[i, j] = 0
            calculated += 1
            if calculated % 50 == 0:
                print(f"计算进度: {calculated}/{total_points}")
    
    plt.figure(figsize=(12, 9))
    
    # 创建等高线图
    contour_lines = plt.contour(T_grid, H_grid, Pp_grid, levels=15, colors='black', alpha=0.5, linewidths=0.8)
    contour_filled = plt.contourf(T_grid, H_grid, Pp_grid, levels=50, cmap='viridis', alpha=0.8)
    
    # 添加标签
    plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')
    
    # 添加颜色条
    cbar = plt.colorbar(contour_filled)
    cbar.set_label('Pp值', fontsize=12)
    
    # 叠加优化轨迹
    if len(optimization_history) > 0:
        df = pd.DataFrame(optimization_history)
        # 筛选SC接近固定值的点
        mask = abs(df['SC'] - SC_fixed) < 0.02
        if mask.sum() > 5:  # 如果有足够的点
            trajectory_df = df[mask].sort_values('iteration')
            plt.plot(trajectory_df['T'], trajectory_df['H'], 'red', linewidth=2, alpha=0.8, label='优化轨迹')
            plt.scatter(trajectory_df['T'], trajectory_df['H'], c='red', s=30, zorder=5)
            
            # 标记起点和终点
            plt.scatter(trajectory_df['T'].iloc[0], trajectory_df['H'].iloc[0], 
                       c='lime', s=100, marker='o', label='起点', zorder=6)
            plt.scatter(trajectory_df['T'].iloc[-1], trajectory_df['H'].iloc[-1], 
                       c='red', s=150, marker='*', label='终点', zorder=6)
            plt.legend()
    
    plt.xlabel('温度 T (K)', fontsize=12)
    plt.ylabel('湿度 H (%)', fontsize=12)
    plt.title(f'目标函数等高线图 (SC = {SC_fixed})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
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
    
    print("开始运行优化算法...")
    print("已优化设置：减少迭代次数，降低求解精度，添加进度显示")
    
    try:
        result = run_optimization_with_visualization()
        
        if len(optimization_history) < 10:
            print("警告：优化数据太少，可能存在问题")
            return result
        
        # 显示结果摘要
        show_optimization_summary()
        
        print("\n正在生成可视化图表...")
        print("将依次显示4个关键图表，每个图单独展示")
        
        # 依次显示关键图表
        print("\n1. 显示收敛过程图...")
        plot_convergence()
        
        input("按Enter键继续查看下一个图表...")
        
        print("\n2. 显示3D参数空间探索图...")
        plot_parameter_space()
        
        input("按Enter键继续查看下一个图表...")
        
        print("\n3. 显示参数演化图...")
        plot_parameter_evolution()
        
        input("按Enter键继续查看下一个图表...")
        
        print("\n4. 显示等高线图...")
        plot_contour_map()
        
        print("\n所有可视化完成！")
        
        return result
    
    except KeyboardInterrupt:
        print("\n用户中断了计算")
        if len(optimization_history) > 0:
            print("正在显示已有的数据...")
            show_optimization_summary()
            plot_convergence()
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 运行主函数
if __name__ == "__main__":
    result = main()