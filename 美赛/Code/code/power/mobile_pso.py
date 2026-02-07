"""
2026 MCM Problem A：智能手机电池消耗建模 — 连续时间模型（数据驱动功率模型）

功率公式（从实测数据拟合，无基础功耗）：
  P_total = P_screen + P_cpu + P_network + P_gps
  P_screen = screen_status × (C1 + C2 × bright_level)
  P_cpu = C3 × (cpu_usage × freq_norm)   # 使用率 × 频率
  P_net = C4 × throughput + C5 × exp(-λ × wifi_intensity)
  P_gps = C6 × gps_status + C7 × (gps_status × gps_activity²)  # 线性 + 二次项

电池模型（一阶 RC 极化模型）：
  ODE 方程组:
    dS/dt  = -I / (Q0 × 3600)           # SOC 动态
    dUp/dt = -Up / (R1×C1) + I / C1     # 极化电压动态
  端电压:
    Um(t) = Vocv(S(t)) - I(t)×R0 - Up(t)

优化方法：粒子群优化算法 (PSO)

验证：用真实放电段数据驱动 ODE，对比模拟 SOC 与实测 battery_level
"""
import os
import glob
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# ==================== 粒子群优化算法 (PSO) ====================
class ParticleSwarmOptimizer:
    """
    粒子群优化算法 (Particle Swarm Optimization)
    
    用于最小化目标函数 f(x)
    """
    def __init__(self, n_particles=30, n_dimensions=7, bounds=None,
                 w=0.7, c1=1.5, c2=1.5, max_iter=200, tol=1e-6, verbose=True):
        """
        参数:
            n_particles: 粒子数量
            n_dimensions: 参数维度
            bounds: 参数边界 [(lower, upper), ...] 每个维度
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否打印迭代信息
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # 设置边界
        if bounds is None:
            self.lb = np.zeros(n_dimensions)
            self.ub = np.ones(n_dimensions) * 10
        else:
            self.lb = np.array([b[0] for b in bounds])
            self.ub = np.array([b[1] for b in bounds])
    
    def optimize(self, objective_func):
        """
        运行 PSO 优化
        
        参数:
            objective_func: 目标函数 f(x) -> float, x 是参数向量
        
        返回:
            best_position: 最优参数
            best_fitness: 最优适应度值
            history: 迭代历史
        """
        # 初始化粒子位置和速度
        positions = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dimensions))
        velocities = np.random.uniform(
            -(self.ub - self.lb) * 0.1,
            (self.ub - self.lb) * 0.1,
            (self.n_particles, self.n_dimensions)
        )
        
        # 计算初始适应度
        fitness = np.array([objective_func(p) for p in positions])
        
        # 个体最优
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        # 全局最优
        gbest_idx = np.argmin(fitness)
        gbest_position = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        # 迭代历史
        history = {'fitness': [gbest_fitness], 'positions': [gbest_position.copy()]}
        
        # 动态惯性权重参数
        w_max = 0.9
        w_min = 0.4
        
        # 主循环
        for iteration in range(self.max_iter):
            # 动态调整惯性权重（线性递减）
            w = w_max - (w_max - w_min) * iteration / self.max_iter
            
            for i in range(self.n_particles):
                # 随机因子
                r1 = np.random.random(self.n_dimensions)
                r2 = np.random.random(self.n_dimensions)
                
                # 更新速度
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # 速度限制
                v_max = (self.ub - self.lb) * 0.2
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # 更新位置
                positions[i] = positions[i] + velocities[i]
                
                # 边界处理（反弹策略）
                for d in range(self.n_dimensions):
                    if positions[i, d] < self.lb[d]:
                        positions[i, d] = self.lb[d]
                        velocities[i, d] = -velocities[i, d] * 0.5
                    elif positions[i, d] > self.ub[d]:
                        positions[i, d] = self.ub[d]
                        velocities[i, d] = -velocities[i, d] * 0.5
                
                # 计算新适应度
                new_fitness = objective_func(positions[i])
                fitness[i] = new_fitness
                
                # 更新个体最优
                if new_fitness < pbest_fitness[i]:
                    pbest_fitness[i] = new_fitness
                    pbest_positions[i] = positions[i].copy()
                    
                    # 更新全局最优
                    if new_fitness < gbest_fitness:
                        gbest_fitness = new_fitness
                        gbest_position = positions[i].copy()
            
            history['fitness'].append(gbest_fitness)
            history['positions'].append(gbest_position.copy())
            
            # 打印进度
            if self.verbose and (iteration + 1) % 20 == 0:
                print(f"    PSO 迭代 {iteration + 1}/{self.max_iter}: 最优适应度 = {gbest_fitness:.6f}")
            
            # 检查收敛
            if len(history['fitness']) > 10:
                recent = history['fitness'][-10:]
                if max(recent) - min(recent) < self.tol:
                    if self.verbose:
                        print(f"    PSO 收敛于迭代 {iteration + 1}")
                    break
        
        return gbest_position, gbest_fitness, history

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)  # code/
BASE_DIR = os.path.dirname(CODE_DIR)    # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 电池参数（Tremblay 标准参数，Li-Po 3.7V/4.2V）----------
Q0 = 8.0          # 标称容量 8.0 Ah (8000 mAh)
R0 = 0.05         # 内阻 (Ω) - 欧姆内阻
E0 = 3.85         # 电池恒定电压 (V)
k = 0.02          # 极化电压幅度 (V)
A = 0.25          # 指数区幅度 (V)
B = 3.0           # 指数区常数 (Ah)^-1

# ---------- 极化参数（一阶 RC 模型）----------
R1 = 0.03         # 极化电阻 (Ω)
C1 = 1000.0       # 极化电容 (F)，时间常数 τ = R1 * C1 = 30s


def estimate_battery_capacity(segment):
    """从放电段数据估计电池容量"""
    df = segment['data']
    
    # 计算总能量消耗
    df_sorted = df.sort_values('timestamp')
    t0 = pd.to_datetime(df_sorted['timestamp'].iloc[0])
    t1 = pd.to_datetime(df_sorted['timestamp'].iloc[-1])
    duration_hours = (t1 - t0).total_seconds() / 3600
    
    # 平均功率
    avg_power = df['battery_power'].abs().mean()
    
    # 总能量 (Wh)
    total_energy_wh = avg_power * duration_hours
    
    # SOC 变化
    soc_drop = segment['soc_drop'] / 100.0  # 转为小数
    
    # 估计容量: Q = E / (U * ΔS)
    # 假设平均电压 3.9V
    avg_voltage = 3.9
    estimated_capacity = total_energy_wh / (avg_voltage * soc_drop)
    
    return estimated_capacity, avg_power, duration_hours, total_energy_wh


def vocv(S, Q=None):
    """开路电压 Vocv(S)"""
    S_safe = np.clip(S, 1e-6, 1.0)
    Q_val = Q if Q is not None else (Q0 if Q0 is not None else 3.0)
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q_val)


# ==================== 1. 加载数据 ====================
def load_mobile_data_for_fitting(max_files=15):
    """加载多个日期文件夹的 mobile data"""
    all_dfs = []
    folders = sorted([d for d in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()])[:max_files]
    
    for folder in folders:
        pattern = os.path.join(DATA_DIR, folder, "*", "*_dynamic_processed.csv")
        files = glob.glob(pattern)
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df['source_file'] = f  # 记录来源
                all_dfs.append(df)
                print(f"  加载: {folder} ({len(df)} 行)")
            except Exception as e:
                print(f"  跳过: {folder} - {e}")
    
    if not all_dfs:
        return None
    data = pd.concat(all_dfs, ignore_index=True)
    print(f"总计加载 {len(data)} 行数据")
    return data


# ==================== 2. 找连续放电段 ====================
def find_discharge_segments(data, min_duration_sec=1800, min_soc_drop=10):
    """
    从数据中找连续放电段
    
    条件：
    - battery_charging_status == 3 (放电)
    - 持续时间 >= min_duration_sec
    - SOC 下降 >= min_soc_drop %
    """
    segments = []
    
    # 按来源文件分组
    for source, group in data.groupby('source_file'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        group['timestamp'] = pd.to_datetime(group['timestamp'])
        
        # 找放电状态
        is_discharge = (group['battery_charging_status'] == 3).values
        
        # 找连续放电段
        seg_start = None
        for i in range(len(group)):
            if is_discharge[i]:
                if seg_start is None:
                    seg_start = i
            else:
                if seg_start is not None:
                    # 段结束，检查是否满足条件
                    seg_end = i
                    seg_df = group.iloc[seg_start:seg_end]
                    duration = (seg_df['timestamp'].iloc[-1] - seg_df['timestamp'].iloc[0]).total_seconds()
                    soc_drop = seg_df['battery_level'].iloc[0] - seg_df['battery_level'].iloc[-1]
                    
                    if duration >= min_duration_sec and soc_drop >= min_soc_drop:
                        segments.append({
                            'data': seg_df.copy(),
                            'source': source,
                            'duration_min': duration / 60,
                            'soc_start': seg_df['battery_level'].iloc[0],
                            'soc_end': seg_df['battery_level'].iloc[-1],
                            'soc_drop': soc_drop
                        })
                    seg_start = None
        
        # 处理最后一段
        if seg_start is not None:
            seg_df = group.iloc[seg_start:]
            duration = (seg_df['timestamp'].iloc[-1] - seg_df['timestamp'].iloc[0]).total_seconds()
            soc_drop = seg_df['battery_level'].iloc[0] - seg_df['battery_level'].iloc[-1]
            if duration >= min_duration_sec and soc_drop >= min_soc_drop:
                segments.append({
                    'data': seg_df.copy(),
                    'source': source,
                    'duration_min': duration / 60,
                    'soc_start': seg_df['battery_level'].iloc[0],
                    'soc_end': seg_df['battery_level'].iloc[-1],
                    'soc_drop': soc_drop
                })
    
    # 按 SOC 下降量排序
    segments.sort(key=lambda x: x['soc_drop'], reverse=True)
    return segments


# ==================== 3. 拟合功率模型（PSO 粒子群优化）====================
def fit_power_model(data, pso_particles=30, pso_max_iter=150):
    """
    使用粒子群优化算法 (PSO) 拟合功率模型
    
    策略：
    1. 按 source_file 分组，每个实验单独用 PSO 拟合一组系数
    2. 以样本量为权重，计算加权平均系数
    3. 这样避免了不同 SOC 范围的数据混淆
    
    参数:
        data: 数据 DataFrame
        pso_particles: PSO 粒子数量
        pso_max_iter: PSO 最大迭代次数
    """
    LAMBDA = 0.03  # WiFi 信号衰减系数
    coef_names = ['C1_screen', 'C2_bright', 'C3_cpu_usage_freq', 'C4_throughput', 'C5_wifi', 'C6_gps', 'C7_gps_quad']
    n_params = len(coef_names)
    
    # 参数边界: 所有系数非负，上界设为合理范围
    param_bounds = [
        (0, 5.0),   # C1_screen: 屏幕基础功率
        (0, 5.0),   # C2_bright: 亮度功率系数
        (0, 10.0),  # C3_cpu_usage_freq: CPU 功率系数
        (0, 3.0),   # C4_throughput: 网络吞吐量功率系数
        (0, 2.0),   # C5_wifi: WiFi 功率系数
        (0, 3.0),   # C6_gps: GPS 线性功率系数
        (0, 3.0),   # C7_gps_quad: GPS 二次项功率系数
    ]
    
    # 存储每个实验的拟合结果
    experiment_results = []
    
    print(f"\n使用粒子群优化算法 (PSO):")
    print(f"  粒子数: {pso_particles}, 最大迭代: {pso_max_iter}")
    print("-" * 60)
    
    # 按来源文件分组
    exp_idx = 0
    for source_file, group_df in data.groupby('source_file'):
        # 只保留放电数据
        df = group_df[group_df['battery_charging_status'] == 3].copy()
        if len(df) < 50:  # 样本太少，跳过
            continue
        
        df['power_abs'] = df['battery_power'].abs()
        df = df[(df['power_abs'] > 0.05) & (df['power_abs'] < 15)]
        if len(df) < 30:
            continue
        
        # 特征构建
        df['screen_on'] = (df['screen_status'] == 1).astype(float)
        df['bright_norm'] = df['bright_level'].clip(0, 255) / 255.0
        df['x_screen_base'] = df['screen_on']
        df['x_screen_bright'] = df['screen_on'] * df['bright_norm']
        
        # CPU: 使用率 × 频率
        cpu_usage_norm = df['cpu_usage'].clip(0, 100) / 100.0
        freq_cols = [c for c in df.columns if c.startswith('frequency_core')]
        if freq_cols:
            freq_norm = df[freq_cols].sum(axis=1) / (3000.0 * len(freq_cols))
        else:
            freq_norm = 0.0
        df['x_cpu_usage_freq'] = cpu_usage_norm * freq_norm
        
        throughput_cols = ['wifi_rx', 'wifi_tx', 'mobile_rx', 'mobile_tx']
        existing_tp = [c for c in throughput_cols if c in df.columns]
        if existing_tp:
            df['x_throughput'] = df[existing_tp].sum(axis=1).clip(0, 1e7) / 1e6
        else:
            df['x_throughput'] = 0.0
        
        if 'wifi_intensity' in df.columns:
            wifi_on = (df['wifi_status'] == 1).astype(float)
            wifi_int = df['wifi_intensity'].clip(-100, -20).fillna(-50)
            df['x_wifi_exp'] = wifi_on * np.exp(-LAMBDA * wifi_int)
        else:
            df['x_wifi_exp'] = 0.0
        
        # GPS 特征：线性 + 二次项 (gps_status × gps_activity²)
        if 'gps_status' in df.columns:
            gps_on = (df['gps_status'] == 1).astype(float)
            df['x_gps'] = gps_on
            if 'gps_activity' in df.columns:
                gps_act_norm = df['gps_activity'].clip(0, 100).fillna(0) / 100.0
                df['x_gps_quad'] = gps_on * (gps_act_norm ** 2)
            else:
                df['x_gps_quad'] = 0.0
        else:
            df['x_gps'] = 0.0
            df['x_gps_quad'] = 0.0
        
        feature_names = ['x_screen_base', 'x_screen_bright', 'x_cpu_usage_freq', 
                         'x_throughput', 'x_wifi_exp', 'x_gps', 'x_gps_quad']
        
        X = df[feature_names].values
        y = df['power_abs'].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        if len(y) < 30:
            continue
        
        exp_idx += 1
        exp_name = os.path.basename(source_file)[:35]
        print(f"\n[{exp_idx}] 拟合实验: {exp_name} ({len(y)} 样本)")
        
        # 使用 PSO 拟合
        X_fit = X
        
        def objective(params):
            """目标函数: 均方误差 (MSE)"""
            y_pred = X_fit @ params
            mse = np.mean((y - y_pred) ** 2)
            return mse
        
        try:
            # 创建 PSO 优化器
            pso = ParticleSwarmOptimizer(
                n_particles=pso_particles,
                n_dimensions=n_params,
                bounds=param_bounds,
                w=0.7,
                c1=1.5,
                c2=1.5,
                max_iter=pso_max_iter,
                tol=1e-8,
                verbose=False  # 减少输出
            )
            
            # 运行优化
            best_params, best_fitness, history = pso.optimize(objective)
            params = best_params
            
            y_pred = X_fit @ params
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # SOC 范围
            soc_range = (df['battery_level'].min(), df['battery_level'].max())
            
            # 预测耗电时长：100% -> 0%，T = (Q0 × U_avg × 1.0) / P_avg
            P_model_avg = np.mean(y_pred)
            T_predict_hours = (Q0 * 3.9 * 1.0) / max(P_model_avg, 0.1)
            
            # 实际实验时长（秒）
            ts = pd.to_datetime(df['timestamp'])
            duration_actual_h = (ts.max() - ts.min()).total_seconds() / 3600
            
            # 记录 PSO 迭代次数
            pso_iterations = len(history['fitness'])
            
            print(f"    PSO 完成: {pso_iterations} 迭代, R2={r2:.4f}, RMSE={rmse:.3f}W")
            
            experiment_results.append({
                'source': os.path.basename(source_file),
                'params': params,
                'n_samples': len(y),
                'r2': r2,
                'rmse': rmse,
                'soc_range': soc_range,
                'P_model_avg': P_model_avg,
                'T_predict_h': T_predict_hours,
                'duration_actual_h': duration_actual_h,
                'pso_iterations': pso_iterations,
                'pso_final_fitness': best_fitness
            })
        except Exception as e:
            print(f"    PSO 拟合失败: {e}")
            continue
    
    if not experiment_results:
        print("没有足够的数据进行拟合")
        return None
    
    # 打印每个实验的结果及预测耗电时长
    print(f"\n" + "=" * 105)
    print(f"PSO 拟合完成: {len(experiment_results)} 个独立实验")
    print("-" * 105)
    print(f"  {'#':<3} {'实验文件':<42} {'n':>5} {'R2':>6} {'SOC':>10} {'P_avg(W)':>9} {'预测时长':>10} {'实际时长':>8}")
    print("-" * 105)
    for i, res in enumerate(experiment_results):
        soc_str = f"{res['soc_range'][0]:.0f}%-{res['soc_range'][1]:.0f}%"
        T_pred = res['T_predict_h']
        T_actual = res['duration_actual_h']
        T_pred_str = f"{T_pred:.1f}h" if T_pred < 24 else f"{T_pred/24:.1f}d"
        T_actual_str = f"{T_actual:.1f}h"
        print(f"  {i+1:<3} {res['source'][:40]:<42} {res['n_samples']:>5} {res['r2']:>6.3f} {soc_str:>10} "
              f"{res['P_model_avg']:>9.2f} {T_pred_str:>10} {T_actual_str:>8}")
    print("-" * 105)
    
    # 加权平均（以样本量为权重）
    total_samples = sum(r['n_samples'] for r in experiment_results)
    weighted_params = np.zeros(n_params)
    for res in experiment_results:
        weight = res['n_samples'] / total_samples
        weighted_params += weight * res['params']
    
    # 计算系数的标准差（反映实验间的差异）
    params_array = np.array([r['params'] for r in experiment_results])
    params_std = np.std(params_array, axis=0)
    
    # 汇总结果
    avg_r2 = np.mean([r['r2'] for r in experiment_results])
    avg_rmse = np.mean([r['rmse'] for r in experiment_results])
    
    coef_dict = dict(zip(coef_names, weighted_params))
    coef_std_dict = dict(zip(coef_names, params_std))
    
    print(f"\n加权平均系数 (PSO, 基于 {len(experiment_results)} 个实验, {total_samples} 个样本):")
    print(f"  平均 R2 = {avg_r2:.4f}, 平均 RMSE = {avg_rmse:.4f} W")
    print("  系数 (均值 ± 标准差):")
    for name in coef_names:
        print(f"    {name}: {coef_dict[name]:.4f} ± {coef_std_dict[name]:.4f}")
    
    return {
        'params': weighted_params, 
        'lambda': LAMBDA, 
        'r2': avg_r2, 
        'rmse': avg_rmse, 
        'coef_dict': coef_dict,
        'coef_std': coef_std_dict,
        'n_experiments': len(experiment_results),
        'experiment_results': experiment_results,
        'optimization_method': 'PSO'
    }


# ==================== 4. 功率计算 ====================
class PowerModel:
    def __init__(self, fit_result):
        self.params = fit_result['params']
        self.lam = fit_result['lambda']
    
    def calc_power_from_row(self, row, return_components=False):
        """从数据行计算功率"""
        screen_on = 1.0 if row.get('screen_status', 0) == 1 else 0.0
        bright_norm = min(max(row.get('bright_level', 0), 0), 255) / 255.0
        
        # CPU: 使用率 × 频率
        cpu_usage = min(max(row.get('cpu_usage', 0), 0), 100) / 100.0  # 0~100
        freq_cols = [c for c in row.index if c.startswith('frequency_core')]
        freq_norm = sum(row.get(c, 0) for c in freq_cols) / (3000.0 * max(len(freq_cols), 1)) if freq_cols else 0.0
        cpu_usage_freq = cpu_usage * freq_norm
        
        throughput = sum(row.get(c, 0) for c in ['wifi_rx', 'wifi_tx', 'mobile_rx', 'mobile_tx']) / 1e6
        throughput = min(max(throughput, 0), 10)
        
        wifi_on = 1.0 if row.get('wifi_status', 0) == 1 else 0.0
        wifi_int = min(max(row.get('wifi_intensity', -50), -100), -20)
        wifi_exp = wifi_on * np.exp(-self.lam * wifi_int)
        
        # GPS: 线性 + 二次项
        gps_on = 1.0 if row.get('gps_status', 0) == 1 else 0.0
        gps_act_norm = min(max(row.get('gps_activity', 0), 0), 100) / 100.0
        gps_quad = gps_on * (gps_act_norm ** 2)
        
        features = [screen_on, screen_on * bright_norm, cpu_usage_freq, throughput, wifi_exp, gps_on, gps_quad]
        
        if return_components:
            # 返回各组件功率（无基础功耗）
            P_screen = self.params[0] * screen_on + self.params[1] * screen_on * bright_norm
            P_cpu = self.params[2] * cpu_usage_freq
            P_network = self.params[3] * throughput + self.params[4] * wifi_exp
            P_gps = self.params[5] * gps_on + self.params[6] * gps_quad
            return {
                'base': 0.0,
                'screen': P_screen,
                'cpu': P_cpu,
                'network': P_network,
                'gps': P_gps,
                'total': max(P_screen + P_cpu + P_network + P_gps, 0.1)
            }
        else:
            P = sum(p * f for p, f in zip(self.params, features))
            return max(P, 0.1)


# ==================== 5. 绘制所有实验 SOC 预测曲线（延伸到 0%）====================
def get_soc_curve_to_empty(segment, power_model, Q_est=None):
    """运行 ODE 仿真，延伸功率直到 SOC 降至 0%"""
    Q_use = Q_est if Q_est is not None else (Q0 if Q0 is not None else 3.0)
    df = segment['data'].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = (df['timestamp'] - t0).dt.total_seconds()
    df['P_model'] = df.apply(power_model.calc_power_from_row, axis=1)
    
    t_data = df['t_sec'].values
    P_data = df['P_model'].values
    P_avg = np.mean(P_data)
    
    # 功率函数：数据段内插值，超出段用平均功率
    P_interp = interp1d(t_data, P_data, kind='linear', fill_value=P_avg, bounds_error=False)
    
    SOC_measured = df['battery_level'].values / 100.0
    S0 = SOC_measured[0]
    Up0 = 0.0
    t_end_data = t_data[-1]
    
    # 估算从满电到 0 的总时间，作为延伸上界
    t_extra_sec = 1.5 * (1.0 * Q_use * 3.9 * 3600) / max(P_avg, 0.1)
    t_span = (t_data[0], t_end_data + t_extra_sec)
    
    def event_soc_zero(t, y):
        return y[0] - 0.01
    event_soc_zero.terminal = True
    event_soc_zero.direction = -1
    
    def odefun(t, y):
        S_val = np.clip(y[0], 1e-6, 1.0)
        Up_val = y[1]
        if S_val <= 1e-6:
            return np.array([0.0, 0.0])
        P = float(P_interp(t))
        Vocv_val = vocv(S_val, Q_use)
        V_eff = Vocv_val - Up_val
        discriminant = V_eff**2 - 4 * R0 * P
        I_val = (V_eff - np.sqrt(max(discriminant, 0))) / (2 * R0) if discriminant >= 0 else V_eff / (2 * R0)
        I_val = max(I_val, 0.0)
        dS_dt = -I_val / (Q_use * 3600.0)
        dUp_dt = -Up_val / (R1 * C1) + I_val / C1
        return np.array([dS_dt, dUp_dt])
    
    dt_median = np.median(np.diff(t_data)) if len(t_data) > 1 else 60
    max_step = min(max(dt_median * 2, 10), 60)
    
    sol = solve_ivp(odefun, t_span, [S0, Up0], method='RK45', dense_output=True,
                    max_step=max_step, events=event_soc_zero)
    
    t_sol = sol.t
    soc_sol = np.clip(sol.sol(t_sol)[0], 0, 1) * 100
    
    return t_sol / 60, soc_sol


def plot_all_experiment_soc_curves(fit_result, data):
    """绘制所有实验的 SOC 下降预测曲线（延伸到 0%）"""
    segments = find_discharge_segments(data, min_duration_sec=600, min_soc_drop=5)
    
    segs_by_basename = defaultdict(list)
    for seg in segments:
        bn = os.path.basename(seg['source'])
        segs_by_basename[bn].append(seg)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 31))
    
    for i, res in enumerate(fit_result['experiment_results']):
        source_bn = res['source']
        if source_bn not in segs_by_basename:
            continue
        segs = segs_by_basename[source_bn]
        best_seg = max(segs, key=lambda s: s['soc_drop'])
        pm = PowerModel({'params': res['params'], 'lambda': fit_result['lambda']})
        
        try:
            t_min, soc_pct = get_soc_curve_to_empty(best_seg, pm, Q_est=Q0)
        except Exception:
            continue
        
        label = f"{i+1}. {source_bn[:35]}"
        ax.plot(t_min, soc_pct, alpha=0.7, linewidth=1.5,
                color=colors[i % 20], label=label)
    
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SOC (%)")
    ax.set_title("所有实验 SOC 下降预测曲线（延伸至 0%）", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(OUTPUT_DIR, "all_experiments_soc_prediction.png"), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {OUTPUT_DIR}/all_experiments_soc_prediction.png")
    plt.close(fig)


# ==================== 6. 用真实数据验证 ====================
def validate_with_real_data(segment, power_model, Q_est=None):
    """
    用真实放电段数据驱动 ODE，对比模拟 SOC 与实测
    
    Q_est: 估计的电池容量 (Ah)，如果为 None 则使用全局 Q0
    """
    # 确定使用的电池容量
    Q_use = Q_est if Q_est is not None else (Q0 if Q0 is not None else 3.0)
    
    df = segment['data'].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算相对时间（秒）
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = (df['timestamp'] - t0).dt.total_seconds()
    
    # 计算每个时刻的功率及各组件贡献
    df['P_model'] = df.apply(power_model.calc_power_from_row, axis=1)
    
    # 计算各组件功率
    power_components = df.apply(lambda row: power_model.calc_power_from_row(row, return_components=True), axis=1)
    df['P_base'] = power_components.apply(lambda x: x['base'])
    df['P_screen'] = power_components.apply(lambda x: x['screen'])
    df['P_cpu'] = power_components.apply(lambda x: x['cpu'])
    df['P_network'] = power_components.apply(lambda x: x['network'])
    df['P_gps'] = power_components.apply(lambda x: x['gps'])
    
    # 实测功率（电流 × 电压）
    df['P_measured'] = df['battery_power'].abs()  # 已经是功率，取绝对值
    
    # 创建功率插值函数
    t_data = df['t_sec'].values
    P_data = df['P_model'].values
    P_interp = interp1d(t_data, P_data, kind='linear', fill_value='extrapolate')
    
    # 实测 SOC
    SOC_measured = df['battery_level'].values / 100.0
    
    # ========== ODE 系统（含极化电压动态）==========
    # 状态变量: y = [S, Up]
    #   dS/dt  = -I / (Q * 3600)
    #   dUp/dt = -Up / (R1 * C1) + I / C1
    # 端电压: Um = Vocv(S) - I * R0 - Up
    # 功率约束: P = Um * I
    
    S0 = SOC_measured[0]
    Up0 = 0.0  # 初始极化电压为 0
    t_span = (t_data[0], t_data[-1])
    
    def odefun(t, y):
        S_val = np.clip(y[0], 1e-6, 1.0)
        Up_val = y[1]
        
        if S_val <= 1e-6:
            return np.array([0.0, 0.0])
        
        P = P_interp(t)
        Vocv_val = vocv(S_val, Q_use)
        
        # 求解电流 I：P = I * (Vocv - I * R0 - Up)
        # => R0 * I^2 - (Vocv - Up) * I + P = 0
        V_eff = Vocv_val - Up_val  # 有效开路电压（扣除极化）
        discriminant = V_eff**2 - 4 * R0 * P
        
        if discriminant < 0:
            # 功率过大，限制电流
            I_val = V_eff / (2 * R0)
        else:
            # 取较小的根（合理的电流值）
            I_val = (V_eff - np.sqrt(discriminant)) / (2 * R0)
        
        I_val = max(I_val, 0.0)  # 电流不能为负
        
        # ODE 方程组（使用 Q_use 而不是全局 Q0）
        dS_dt = -I_val / (Q_use * 3600.0)
        dUp_dt = -Up_val / (R1 * C1) + I_val / C1
        
        return np.array([dS_dt, dUp_dt])
    
    # 动态设置 max_step：取数据采样间隔的 2 倍
    dt_median = np.median(np.diff(t_data)) if len(t_data) > 1 else 60
    max_step = min(max(dt_median * 2, 10), 60)  # 限制在 10~60 秒
    sol = solve_ivp(odefun, t_span, [S0, Up0], method='RK45', dense_output=True, max_step=max_step)
    
    # 在数据时间点评估
    sol_values = sol.sol(t_data)
    SOC_simulated = np.clip(sol_values[0], 0, 1)
    Up_simulated = sol_values[1]  # 极化电压
    
    # 计算误差
    error = SOC_simulated - SOC_measured
    rmse = np.sqrt(np.mean(error**2)) * 100  # 转为百分比
    mae = np.mean(np.abs(error)) * 100
    
    return {
        't_min': t_data / 60,
        'SOC_measured': SOC_measured * 100,
        'SOC_simulated': SOC_simulated * 100,
        'Up_simulated': Up_simulated,  # 极化电压 (V)
        'P_model': P_data,
        'P_measured': df['P_measured'].values,
        'P_base': df['P_base'].values,
        'P_screen': df['P_screen'].values,
        'P_cpu': df['P_cpu'].values,
        'P_network': df['P_network'].values,
        'P_gps': df['P_gps'].values,
        'rmse_pct': rmse,
        'mae_pct': mae,
        'Q_used': Q_use,  # 使用的电池容量 (Ah)
        'df': df
    }


# ==================== 主程序 ====================
print("=" * 60)
print("从 mobile data 拟合功率模型 + 真实数据验证")
print("=" * 60)

data = load_mobile_data_for_fitting(max_files=20)

if data is not None:
    # 拟合功率模型
    fit_result = fit_power_model(data)
    
    if fit_result is not None:
        power_model = PowerModel(fit_result)
        
        # 绘制所有实验 SOC 预测曲线
        plot_all_experiment_soc_curves(fit_result, data)
        
        # 保存拟合参数及预测耗电时长
        discharge_predictions = [
            {"experiment": r['source'], "P_avg_W": float(r['P_model_avg']),
             "T_predict_h": float(r['T_predict_h']), "duration_actual_h": float(r['duration_actual_h']),
             "soc_range": [float(r['soc_range'][0]), float(r['soc_range'][1])], "R2": float(r['r2'])}
            for r in fit_result['experiment_results']
        ]
        with open(os.path.join(OUTPUT_DIR, "power_model_pso.json"), "w", encoding="utf-8") as f:
            json.dump({
                "method": "PSO (Particle Swarm Optimization)",
                "optimization": {
                    "algorithm": "粒子群优化算法",
                    "particles": 30,
                    "max_iterations": 150
                },
                "n_experiments": fit_result['n_experiments'],
                "battery_parameters": {
                    "Q0_Ah": float(Q0),
                    "R0_ohm": float(R0),
                    "R1_ohm": float(R1),
                    "C1_F": float(C1),
                    "tau_s": float(R1 * C1)
                },
                "coefficients": {k: float(v) for k, v in fit_result['coef_dict'].items()},
                "coefficients_std": {k: float(v) for k, v in fit_result['coef_std'].items()},
                "lambda": fit_result['lambda'],
                "avg_R2": fit_result['r2'],
                "avg_RMSE": fit_result['rmse'],
                "discharge_predictions": discharge_predictions
            }, f, indent=2, ensure_ascii=False)
        
        # 找连续放电段
        print("\n查找连续放电段 (时长 ≥ 30min, SOC下降 ≥ 10%)...")
        segments = find_discharge_segments(data)
        print(f"找到 {len(segments)} 个符合条件的放电段")
        
        if segments:
            # 显示前几个放电段
            print("\n最佳放电段:")
            for i, seg in enumerate(segments[:5]):
                print(f"  {i+1}. SOC: {seg['soc_start']:.0f}% -> {seg['soc_end']:.0f}% "
                      f"(下降 {seg['soc_drop']:.0f}%), 时长 {seg['duration_min']:.0f} min")
            
            # 选择最佳放电段进行验证
            best_seg = segments[0]
            print(f"\n选择放电段: SOC {best_seg['soc_start']:.0f}% -> {best_seg['soc_end']:.0f}%, "
                  f"时长 {best_seg['duration_min']:.0f} min")
            
            # 估计电池容量（仅供参考）
            Q_est, avg_pwr, duration_h, total_energy = estimate_battery_capacity(best_seg)
            print(f"\n电池容量信息:")
            print(f"  固定容量: {Q0:.2f} Ah ({Q0*1000:.0f} mAh)")
            print(f"  数据估计: {Q_est:.2f} Ah ({Q_est*1000:.0f} mAh)")
            print(f"  平均实测功率: {avg_pwr:.2f} W, 放电时长: {duration_h:.2f} h")
            
            # 验证（使用固定的 Q0）
            result = validate_with_real_data(best_seg, power_model, Q_est=Q0)
            
            print(f"\n验证结果（使用固定容量 {result['Q_used']:.2f} Ah）:")
            print(f"  SOC RMSE: {result['rmse_pct']:.2f}%")
            print(f"  SOC MAE:  {result['mae_pct']:.2f}%")
            
            # 诊断：功率对比
            P_model_avg = np.mean(result['P_model'])
            P_measured_avg = np.mean(result['P_measured'])
            P_rmse = np.sqrt(np.mean((result['P_model'] - result['P_measured'])**2))
            print(f"\n功率诊断:")
            print(f"  模型功率: 平均 {P_model_avg:.2f} W, 范围 [{np.min(result['P_model']):.2f}, {np.max(result['P_model']):.2f}]")
            print(f"  实测功率: 平均 {P_measured_avg:.2f} W, 范围 [{np.min(result['P_measured']):.2f}, {np.max(result['P_measured']):.2f}]")
            print(f"  功率 RMSE: {P_rmse:.2f} W")
            
            # 计算基于平均功率的理论 SOC 下降
            duration_sec = result['t_min'][-1] * 60
            theoretical_soc_drop = (P_model_avg / 3.9) / (result['Q_used'] * 3600) * duration_sec * 100
            actual_soc_drop = result['SOC_measured'][0] - result['SOC_measured'][-1]
            print(f"  理论 SOC 下降（基于模型平均功率）: {theoretical_soc_drop:.1f}%")
            print(f"  实际 SOC 下降: {actual_soc_drop:.1f}%")
            
            # 极化电压诊断
            Up_avg = np.mean(result['Up_simulated'])
            Up_max = np.max(result['Up_simulated'])
            print(f"\n极化电压诊断:")
            print(f"  极化电压: 平均 {Up_avg*1000:.1f} mV, 最大 {Up_max*1000:.1f} mV")
            print(f"  极化参数: R1={R1:.3f} Ω, C1={C1:.0f} F, τ={R1*C1:.1f} s")
            
            # ==================== 绘图 ====================
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            
            # ① SOC 对比
            ax = axes[0, 0]
            ax.plot(result['t_min'], result['SOC_measured'], 'b-', linewidth=2, label='实测 SOC')
            ax.plot(result['t_min'], result['SOC_simulated'], 'r--', linewidth=2, label='模拟 SOC')
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("SOC (%)")
            ax.set_title(f"① SOC 对比 (RMSE={result['rmse_pct']:.2f}%)", fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.4)
            
            # ② SOC 误差 + 极化电压
            ax = axes[0, 1]
            error = result['SOC_simulated'] - result['SOC_measured']
            ax.plot(result['t_min'], error, 'g-', linewidth=1.5, label='SOC误差')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax.fill_between(result['t_min'], error, 0, alpha=0.2, color='green')
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("SOC 误差 (%)", color='green')
            ax.tick_params(axis='y', labelcolor='green')
            
            # 双Y轴：极化电压
            ax2 = ax.twinx()
            ax2.plot(result['t_min'], result['Up_simulated'] * 1000, 'm-', linewidth=1.5, alpha=0.8, label='极化电压')
            ax2.set_ylabel("极化电压 Up (mV)", color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            ax.set_title("② SOC误差 & 极化电压动态", fontsize=11)
            ax.grid(True, alpha=0.4)
            
            # ③ 功率对比（实测 vs 模型）
            ax = axes[1, 0]
            ax.plot(result['t_min'], result['P_measured'], 'b-', linewidth=1, alpha=0.6, label='实测功率')
            ax.plot(result['t_min'], result['P_model'], 'r-', linewidth=1.5, label='模型功率')
            P_rmse = np.sqrt(np.mean((result['P_model'] - result['P_measured'])**2))
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"③ 功率对比 (RMSE={P_rmse:.2f}W)", fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.4)
            
            # ④ 各组件功率贡献（堆叠面积图，无基础功耗）
            ax = axes[1, 1]
            
            # 计算累积功率
            P_cum1 = result['P_screen']
            P_cum2 = P_cum1 + result['P_cpu']
            P_cum3 = P_cum2 + result['P_network']
            P_cum4 = P_cum3 + result['P_gps']
            
            # 堆叠面积图
            ax.fill_between(result['t_min'], 0, P_cum1,
                           label='屏幕', color='#ffa726', alpha=0.7)
            ax.fill_between(result['t_min'], P_cum1, P_cum2,
                           label='CPU', color='#66bb6a', alpha=0.7)
            ax.fill_between(result['t_min'], P_cum2, P_cum3,
                           label='网络', color='#42a5f5', alpha=0.7)
            ax.fill_between(result['t_min'], P_cum3, P_cum4,
                           label='GPS', color='#ef5350', alpha=0.7)
            
            # 总功率线
            ax.plot(result['t_min'], result['P_model'], 'k-', linewidth=1.5, label='总功率')
            
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Power (W)")
            ax.set_title("④ 各组件功率贡献", fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            fig.suptitle(f"真实数据验证: SOC {best_seg['soc_start']:.0f}% -> {best_seg['soc_end']:.0f}%, "
                        f"时长 {best_seg['duration_min']:.0f} min, 电池容量 {result['Q_used']:.1f} Ah", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "mobile_power_validation.png"), dpi=150, bbox_inches="tight")
            print(f"\nFigure saved: {OUTPUT_DIR}/mobile_power_validation.png")
            
            plt.close("all")
        else:
            print("未找到符合条件的放电段")
        
        print("\nDone.")
    else:
        print("功率模型拟合失败")
else:
    print("数据加载失败")
