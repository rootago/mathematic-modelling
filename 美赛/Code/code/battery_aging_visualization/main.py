"""
电池老化可视化
模拟不同使用模式和温度下，单次放电过程中电池容量的动态变化
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 添加父目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USAGE_PATTERNS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "usage_patterns")
sys.path.insert(0, USAGE_PATTERNS_DIR)

from predict_power import (  # type: ignore
    load_power_model,
    load_mobile_data_stats,
    table_to_row,
    calc_power,
    mix_to_sequence,
    USAGE_PATTERNS,
    Q0,
    V_avg,
    NAME_EN,
)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 三种温度条件
TEMPERATURES = [10, 25, 40]
TEMP_LABELS = {10: "10°C", 25: "25°C", 40: "40°C"}

# 电池参数（从mobile_new.py）
R0 = 0.05
R1 = 0.03
C1 = 1000.0
E0 = 4.1
k = 0.01
A = 0.25
B = 3.0


def soc_indicator(S):
    """SOC指示函数：S>95%或S<10%时返回1"""
    S_pct = S * 100
    return 1.0 if (S_pct > 95 or S_pct < 10) else 0.0


def vocv(S, Q=None):
    """开路电压"""
    S_safe = np.clip(S, 1e-6, 1.0)
    Q_val = Q if Q is not None else Q0
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q_val)


def degradation_stress(T, I_current, S, v1, v2):
    """退化应力 D = exp(0.035*(T-25)) * (1 + v1*|I| + v2*I(SOC))"""
    temp_factor = np.exp(0.035 * (T - 25))
    current_factor = 1 + v1 * abs(I_current) + v2 * soc_indicator(S)
    return temp_factor * current_factor


def temperature_degradation_factor(T, theta2):
    """温度衰减因子 y(T) = exp(-θ₂ * T)"""
    return np.exp(-theta2 * T)


def aged_capacity(F, T, theta2, Q_nominal):
    """老化后的容量 Qm = Q0 * F * y(T)"""
    y_T = temperature_degradation_factor(T, theta2)
    return Q_nominal * F * y_T


def simulate_single_discharge(pattern_name, mix_sequence, activity_powers, model, T_ambient):
    """
    模拟单次放电过程（100% -> 接近0%）
    返回 (time_min, SOC_array, F_array, Qm_array)
    """
    theta1 = model["theta1"]
    theta2 = model["theta2"]
    v1 = model["v1"]
    v2 = model["v2"]
    
    # 计算各活动段的信息
    activities = []
    for act, pct in mix_sequence:
        P = activity_powers[act]
        activities.append({'name': act, 'pct': pct, 'power': P})
    
    # 估算总时长（小时）：基于平均功率
    P_avg = sum(act['power'] * act['pct'] for act in activities)
    energy_total_wh = Q0 * V_avg
    estimated_duration_h = energy_total_wh / P_avg
    estimated_duration_min = estimated_duration_h * 60
    
    # ODE系统：dS/dt, dUp/dt, dF/dt
    def ode_system(t, y):
        S = np.clip(y[0], 1e-6, 1.0)
        Up = y[1]
        F = np.clip(y[2], 0.5, 1.0)
        
        if S <= 1e-6:
            return [0.0, 0.0, 0.0]
        
        # 确定当前活动（基于时间进度）
        progress = t / estimated_duration_min  # 0-1
        cumulative = 0.0
        current_P = activities[0]['power']
        for act in activities:
            if progress < cumulative + act['pct']:
                current_P = act['power']
                break
            cumulative += act['pct']
        
        # 计算有效容量
        Qm = aged_capacity(F, T_ambient, theta2, Q0)
        Qm = max(Qm, Q0 * 0.5)
        
        # 计算电流（从功率）
        Vocv_val = vocv(S, Qm)
        V_eff = Vocv_val - Up
        discriminant = V_eff**2 - 4 * R0 * current_P
        
        if discriminant < 0:
            I_val = V_eff / (2 * R0)
        else:
            I_val = (V_eff - np.sqrt(discriminant)) / (2 * R0)
        I_val = max(I_val, 0.0)
        
        # 计算退化应力
        D = degradation_stress(T_ambient, I_val, S, v1, v2)
        
        # 微分方程（注意时间单位：分钟）
        dS_dt = -I_val / (Qm * 3600.0) * 60  # 转换为每分钟
        dUp_dt = (-Up / (R1 * C1) + I_val / C1) * 60
        dF_dt = -theta1 * D * 60
        
        return [dS_dt, dUp_dt, dF_dt]
    
    # 初始条件
    S0 = 1.0
    Up0 = 0.0
    F0 = 1.0
    
    # 时间跨度（分钟）
    t_span = (0, estimated_duration_min)
    t_eval = np.linspace(0, estimated_duration_min, 500)
    
    # 求解ODE
    sol = solve_ivp(ode_system, t_span, [S0, Up0, F0], 
                    t_eval=t_eval, method='RK45', max_step=5.0)
    
    time_min = sol.t
    SOC_values = np.clip(sol.y[0], 0, 1) * 100
    F_values = np.clip(sol.y[2], 0.5, 1.0)
    
    # 计算Qm
    Qm_values = np.array([aged_capacity(F, T_ambient, theta2, Q0) for F in F_values])
    
    return time_min, SOC_values, F_values, Qm_values


def plot_aging_curves_by_temperature(all_results, output_dir):
    """
    为每种温度绘制一张图，包含所有行为模式的Qm曲线
    """
    pattern_names = list(USAGE_PATTERNS.keys())
    
    for temp in TEMPERATURES:
        fig, ax1 = plt.subplots(figsize=(12, 7), facecolor="white")
        ax1.set_facecolor("white")
        
        # 为F创建右侧Y轴
        ax2 = ax1.twinx()
        
        # 绘制每种行为模式的曲线
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        
        for i, pattern_name in enumerate(pattern_names):
            time_min, SOC_values, F_values, Qm_values = all_results[pattern_name][temp]
            time_h = time_min / 60  # 转换为小时
            pattern_label = NAME_EN.get(pattern_name, pattern_name)
            
            # 绘制Qm（红色系，左Y轴）
            ax1.plot(time_h, Qm_values, 
                    color=colors[i], linewidth=2.5, 
                    label=pattern_label, alpha=0.85)
        
        # 设置左Y轴（Qm）
        ax1.set_xlabel("Time (hours)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Effective Capacity Qm (Ah)", fontsize=13, fontweight="bold", color="#c0392b")
        ax1.tick_params(axis='y', labelcolor="#c0392b")
        ax1.set_ylim(Q0 * 0.95, Q0 * 1.01)
        
        # 标题
        ax1.set_title(f"Battery Aging Dynamics at {TEMP_LABELS[temp]} (theta1={model['theta1']:.1e}, theta2={model['theta2']:.1e})", 
                     fontsize=14, fontweight="bold", pad=15)
        
        # 设置右Y轴（F）- 暂时隐藏，只显示Qm
        ax2.set_ylim(0.95, 1.01)
        ax2.set_ylabel("", fontsize=13)
        ax2.tick_params(axis='y', labelcolor="white")
        
        # 网格和图例
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(loc="lower left", framealpha=0.95, fontsize=10)
        
        # 美化
        for spine in ["top"]:
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"aging_curves_{temp}C.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  已保存: {output_path}")


def save_results_to_json(all_results, output_dir):
    """保存结果到JSON"""
    output_data = {}
    
    for pattern_name in all_results:
        output_data[pattern_name] = {}
        for temp in TEMPERATURES:
            time_min, SOC_values, F_values, Qm_values = all_results[pattern_name][temp]
            output_data[pattern_name][f"{temp}C"] = {
                "time_minutes": time_min.tolist(),
                "time_hours": (time_min / 60).tolist(),
                "SOC_percent": SOC_values.tolist(),
                "F_degradation_factor": F_values.tolist(),
                "Qm_ah": Qm_values.tolist()
            }
    
    output_path = os.path.join(output_dir, "aging_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  已保存数据: {output_path}")


def main():
    global model  # 用于标题
    
    print("=" * 70)
    print("电池老化动态模拟 - 单次放电过程")
    print("=" * 70)
    
    # 加载模型
    print("\n加载功率模型...")
    model = load_power_model()
    
    print("加载 mobile data 统计...")
    stats = load_mobile_data_stats()
    
    # 计算各活动的功率
    from predict_power import ACTIVITY_TABLE
    all_acts = set()
    for mix_list in USAGE_PATTERNS.values():
        for act, _ in mix_list:
            all_acts.add(act)
    
    activity_powers = {}
    for act in all_acts:
        row = table_to_row(act, stats)
        activity_powers[act] = calc_power(row, model)
    
    print("\n各活动典型功率 (W):")
    for act, P in activity_powers.items():
        print(f"  {act}: {P:.3f} W")
    
    # 模拟所有模式
    print("\n开始模拟单次放电过程...")
    all_results = {}
    
    for pattern_name, mix_list in USAGE_PATTERNS.items():
        print(f"\n处理模式: {pattern_name}")
        mix_sequence = mix_to_sequence(mix_list)
        
        all_results[pattern_name] = {}
        
        for temp in TEMPERATURES:
            print(f"  温度 {temp}°C...", end=" ")
            time_min, SOC_values, F_values, Qm_values = simulate_single_discharge(
                pattern_name, mix_sequence, activity_powers, model, temp
            )
            all_results[pattern_name][temp] = (time_min, SOC_values, F_values, Qm_values)
            print(f"时长 {time_min[-1]:.0f} min, Qm: {Qm_values[0]:.3f} -> {Qm_values[-1]:.3f} Ah")
    
    # 创建输出目录
    output_dir = SCRIPT_DIR
    
    # 绘制图表
    print("\n绘制曲线...")
    plot_aging_curves_by_temperature(all_results, output_dir)
    
    # 保存数据
    print("\n保存数据...")
    save_results_to_json(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  - aging_curves_10C.png")
    print("  - aging_curves_25C.png")
    print("  - aging_curves_40C.png")
    print("  - aging_data.json")


if __name__ == "__main__":
    main()
