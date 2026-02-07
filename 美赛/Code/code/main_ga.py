"""
2026 MCM Problem A：智能手机电池消耗建模 — 使用遗传算法拟合参数

与 main.py 的区别：使用遗传算法（Genetic Algorithm, GA）代替最小二乘法拟合 k, A, B。

模型假设：锂离子电池（Li-ion）。等效电路 + OCV-SOC 经验关系。
控制方程（连续时间）：
  dS/dt = -I/Q0，I = P_total/U，U = Vocv - I*r，
  Vocv = E0 - k/S + A*exp(-B*(1-S)*Q0)。

========== 数据与处理说明 ==========
- 数据来源：data2（Arbin 格式），需可免费获取、开放许可，报告中注明出处。
- 文件：data/data2/SP2/DST_25C/11_05_2015_SP20-2_DST_50SOC.xls
- 使用的列：Test_Time(s), Current(A), Voltage(V), Discharge_Capacity(Ah)。
- SOC = 1 - Discharge_Capacity(Ah)/Q0，[0,1]。Arbin 放电 I>0，仅保留放电段拟合。
- 拟合：遗传算法 min sum (U_obs - (Vocv(SOC)-I*r))^2，得 k,A,B。
- 图1：U 实测 vs 模型、残差、数据 SOC；图2：ODE 模拟 S(t), U(t), I(t), Vocv(S)。

依赖：pip install deap (可选，会自动回退到简化的遗传算法实现)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 尝试导入 DEAP（专业遗传算法库）
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    print("警告: DEAP 未安装。将使用简化的遗传算法实现。")
    print("如需完整功能，请运行: pip install deap")
    DEAP_AVAILABLE = False

# 中文字体，避免 "Glyph missing" 警告（Windows 常用 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 常数 ----------
Ptotal = 20.0
Q0 = 2.0
r = 0.022
E0 = 3.6

# 工况数据路径（data2 DST 25C 50SOC）
DYNAMIC_DATA_PATH = os.path.join(DATA_DIR, "data2", "SP2", "DST_25C", "11_05_2015_SP20-2_DST_50SOC.xls")


def vocv(S, E0, k, A, B, Q0):
    """Vocv = E0 - k/S + A*exp(-B*(1-S)*Q0)。"""
    S_safe = np.clip(S, 1e-6, 1.0)
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q0)


def load_dynamic_data(file_path, q0=Q0, discharge_only=True, I_discharge_threshold=0.01):
    """
    读 data2 工况数据，返回 t, I, U, S（SOC）。

    使用的列（Arbin/Channel sheet 典型列名）：
      - Test_Time(s)           -> t (s)
      - Current(A)             -> I (A)：Arbin 惯例 放电>0，充电<0
      - Voltage(V)             -> U (V)，端电压
      - Discharge_Capacity(Ah) -> 累计放电容量，用于 SOC

    SOC 定义：SOC = 1 - Discharge_Capacity(Ah) / Q0，范围 [0, 1]。
    discharge_only：若 True，只保留 I > I_discharge_threshold 的放电阶段。
    """
    xl = pd.ExcelFile(file_path)
    for sheet in xl.sheet_names:
        if sheet == "Info":
            continue
        df = pd.read_excel(file_path, sheet_name=sheet)
        cur_c = next((c for c in df.columns if "Current" in c and "A)" in c), None)
        volt_c = next((c for c in df.columns if "Voltage" in c and "V)" in c), None)
        dch_c = next((c for c in df.columns if "Discharge" in c and "Capacity" in c), None)
        time_c = next((c for c in df.columns if "Test_Time" in c), None)
        if time_c is None:
            time_c = next((c for c in df.columns if "Time" in c and "s)" in c), None)
        if cur_c is None or volt_c is None or dch_c is None:
            continue
        t = df[time_c].values.astype(float) if time_c else np.arange(len(df)) * 1.0
        I = df[cur_c].values.astype(float)
        U = df[volt_c].values.astype(float)
        Dch = df[dch_c].values.astype(float)
        # SOC = 1 - Discharge_Capacity/Q0，限制在 [0, 1]
        S = np.clip(1.0 - Dch / q0, 1e-6, 1.0)
        if discharge_only:
            mask = I > I_discharge_threshold
            if np.sum(mask) < 10:
                mask = I >= 0  # 放电点过少时放宽为 I >= 0
            t, I, U, S = t[mask], I[mask], U[mask], S[mask]
        return t, I, U, S
    return None, None, None, None


def u_from_vocv(Vocv, Ptotal, r):
    """U = Vocv - I*r 与 I = Ptotal/U => U^2 - Vocv*U + Ptotal*r = 0，取正根。"""
    delta = np.maximum(Vocv**2 - 4 * Ptotal * r, 0.0)
    return (Vocv + np.sqrt(delta)) / 2.0


# 简化的遗传算法实现（当 DEAP 不可用时使用）
def simple_ga(objective, bounds, pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.2, seed=42):
    """简化的遗传算法实现"""
    np.random.seed(seed)
    n_params = len(bounds)
    
    # 初始化种群
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, n_params)
    )
    
    best_fitness = float('inf')
    best_individual = None
    nfev = 0
    
    print(f"遗传算法：种群大小={pop_size}, 代数={generations}")
    
    for gen in range(generations):
        # 评估适应度
        fitness = np.array([objective(ind) for ind in population])
        nfev += pop_size
        
        # 记录最优个体
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_individual = population[min_idx].copy()
        
        if gen % 20 == 0 or gen == generations - 1:
            print(f"代 {gen:3d}: 最优适应度 = {best_fitness:.6f}")
        
        # 选择（锦标赛选择）
        new_population = []
        for _ in range(pop_size):
            # 锦标赛选择
            tournament_idx = np.random.choice(pop_size, size=3, replace=False)
            winner_idx = tournament_idx[np.argmin(fitness[tournament_idx])]
            new_population.append(population[winner_idx].copy())
        
        new_population = np.array(new_population)
        
        # 交叉
        for i in range(0, pop_size - 1, 2):
            if np.random.random() < crossover_rate:
                # 单点交叉
                cross_point = np.random.randint(1, n_params)
                new_population[i, cross_point:], new_population[i+1, cross_point:] = \
                    new_population[i+1, cross_point:].copy(), new_population[i, cross_point:].copy()
        
        # 变异
        for i in range(pop_size):
            if np.random.random() < mutation_rate:
                # 高斯变异
                mut_idx = np.random.randint(n_params)
                mutation = np.random.normal(0, 0.1 * (bounds[mut_idx][1] - bounds[mut_idx][0]))
                new_population[i, mut_idx] += mutation
                # 边界检查
                new_population[i, mut_idx] = np.clip(
                    new_population[i, mut_idx],
                    bounds[mut_idx][0],
                    bounds[mut_idx][1]
                )
        
        population = new_population
    
    return best_individual, best_fitness, nfev


# ========== 1. 使用遗传算法拟合 k, A, B ==========
if not os.path.exists(DYNAMIC_DATA_PATH):
    print("未找到工况数据:", DYNAMIC_DATA_PATH)
    k, A, B = 0.15, 0.5, 2.0 / Q0
    print("使用默认 k, A, B。")
    t_fit = I_fit = U_obs_fit = S_fit = None
else:
    t_fit, I_fit, U_obs_fit, S_fit = load_dynamic_data(DYNAMIC_DATA_PATH)
    if t_fit is None:
        print("未能解析数据列")
        k, A, B = 0.15, 0.5, 2.0 / Q0
        t_fit = I_fit = U_obs_fit = S_fit = None
    else:
        print("使用的列: Test_Time(s), Current(A), Voltage(V), Discharge_Capacity(Ah); "
              "SOC = 1 - Discharge_Capacity(Ah)/Q0，范围 [0, 1]。")
        print("仅使用放电阶段（Arbin 惯例：Current(A) > 0 为放电，已过滤 I > 0.01 A）。")
        print(f"SOC 范围: 首 {S_fit[0]:.4f} ~ 末 {S_fit[-1]:.4f}")
        step = max(1, len(t_fit) // 2000)
        t_fit = t_fit[::step]
        I_fit = I_fit[::step]
        U_obs_fit = U_obs_fit[::step]
        S_fit = S_fit[::step]
        print(f"遗传算法拟合数据点数: {len(t_fit)}, 时间范围: {t_fit[0]:.1f} ~ {t_fit[-1]:.1f} s")

        # 定义目标函数：残差平方和
        def objective(p):
            kv, Av, Bv = p[0], p[1], p[2]
            V = vocv(S_fit, E0, kv, Av, Bv, Q0)
            U_mod = V - I_fit * r
            residuals = U_obs_fit - U_mod
            return np.sum(residuals**2)  # 返回残差平方和

        # 参数边界
        bounds = [(1e-6, 2.0), (1e-6, 5.0), (1e-6, 10.0)]  # k, A, B

        print("开始遗传算法优化...")
        
        if DEAP_AVAILABLE:
            # 使用 DEAP 的专业遗传算法实现
            # 创建类型
            if hasattr(creator, "FitnessMin"):
                del creator.FitnessMin
            if hasattr(creator, "Individual"):
                del creator.Individual
                
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            toolbox = base.Toolbox()
            
            # 个体生成
            def create_individual():
                return [np.random.uniform(b[0], b[1]) for b in bounds]
            
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # 评估函数
            def eval_func(individual):
                return (objective(individual),)
            
            toolbox.register("evaluate", eval_func)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # 参数边界约束
            def check_bounds(individual):
                for i, (low, high) in enumerate(bounds):
                    individual[i] = np.clip(individual[i], low, high)
                return individual
            
            toolbox.decorate("mate", tools.DeltaPenalty(check_bounds, float('inf')))
            toolbox.decorate("mutate", tools.DeltaPenalty(check_bounds, float('inf')))
            
            # 运行遗传算法
            pop = toolbox.population(n=100)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            
            pop, logbook = algorithms.eaSimple(
                pop, toolbox,
                cxpb=0.7,           # 交叉概率
                mutpb=0.2,          # 变异概率
                ngen=200,           # 代数
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            
            best = hof[0]
            k, A, B = best
            final_cost = objective(best)
            nfev = 100 * 200  # 粗略估计
            method_name = "Genetic Algorithm (DEAP)"
            
        else:
            # 使用简化的遗传算法实现
            best, final_cost, nfev = simple_ga(objective, bounds, pop_size=100, generations=200, seed=42)
            k, A, B = best
            method_name = "Genetic Algorithm (Simple Implementation)"
        
        print(f"\n遗传算法拟合结果: k = {k:.6f}, A = {A:.6f}, B = {B:.6f}, 残差平方和 = {final_cost:.6f}")
        print(f"函数评估次数: {nfev}")
        
        # 保存拟合参数
        with open(os.path.join(OUTPUT_DIR, "vocv_fit_params_ga.json"), "w", encoding="utf-8") as f:
            json.dump({
                "method": method_name,
                "E0": E0, 
                "k": k, 
                "A": A, 
                "B": B, 
                "Q0": Q0,
                "residual_sum_squares": float(final_cost),
                "nfev": int(nfev)
            }, f, indent=2, ensure_ascii=False)

# ========== 2. ODE：dS/dt = -I/Q0，用拟合得到的 k,A,B；S 限制在 [0, 1] ==========
def odefun(t, S):
    S_val = np.clip(S[0], 1e-6, 1.0)
    if S_val <= 1e-6:
        return np.array([0.0])
    Vocv_val = vocv(S_val, E0, k, A, B, Q0)
    U_val = u_from_vocv(Vocv_val, Ptotal, r)
    I_val = Ptotal / (U_val + 1e-12)
    return np.array([-I_val / (Q0 * 3600.0)])

S0 = 0.5
t_span = (0, 3600)
sol = solve_ivp(odefun, t_span, [S0], method="RK45", dense_output=True)
t_plot = np.linspace(0, t_span[1], 300)
S_plot = sol.sol(t_plot)[0]
Vocv_plot = vocv(S_plot, E0, k, A, B, Q0)
U_plot = u_from_vocv(Vocv_plot, Ptotal, r)
I_plot = Ptotal / U_plot

# ========== 3. 绘图 ==========
if t_fit is not None:
    Vocv_fit = vocv(S_fit, E0, k, A, B, Q0)
    U_model_fit = Vocv_fit - I_fit * r
    t_rel_min = (t_fit - t_fit.min()) / 60.0
    
    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes1[0].plot(t_rel_min, U_obs_fit, alpha=0.7, label="U 实测", color="#1565c0")
    axes1[0].plot(t_rel_min, U_model_fit, alpha=0.9, label="U 模型 (Vocv(S)-Ir)", color="#b71c1c", linewidth=1.2)
    axes1[0].set_ylabel(r"Voltage (V)")
    axes1[0].set_xlim(0, None)
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.5)
    axes1[0].set_title("① 端电压：实测 vs 模型（遗传算法拟合）", fontsize=10)
    
    axes1[1].plot(t_rel_min, U_obs_fit - U_model_fit, color="#2e7d32", alpha=0.8)
    axes1[1].set_ylabel(r"$U_{obs} - U_{model}$")
    axes1[1].set_xlim(0, None)
    axes1[1].grid(True, alpha=0.5)
    axes1[1].set_title("② 拟合残差", fontsize=10)
    
    axes1[2].plot(t_rel_min, S_fit, color="#6a1b9a", linewidth=1.5, label=r"数据 SOC")
    axes1[2].set_ylabel(r"SOC")
    axes1[2].set_ylim(0, 1.05)
    axes1[2].set_xlabel(r"Time (min)")
    axes1[2].set_xlim(0, None)
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.5)
    axes1[2].set_title("③ SOC 随时间变化", fontsize=10)
    
    fig1.suptitle("图1：遗传算法拟合结果（数据来源：data2 DST 放电段）", fontsize=12)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "fit_joint_U_ga.png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved: {OUTPUT_DIR}/fit_joint_U_ga.png")

S_plot = np.clip(S_plot, 0.0, 1.0)
fig2, axes2 = plt.subplots(2, 2, figsize=(9, 7))

ax = axes2[0, 0]
ax.plot(t_plot / 60, S_plot, color="#0d47a1", linewidth=2)
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"SOC $S$")
ax.set_ylim(0, 1.05)
ax.set_title("① SOC 随时间", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[0, 1]
ax.plot(t_plot / 60, U_plot, color="#1565c0", linewidth=2, label=r"$U$")
ax.plot(t_plot / 60, Vocv_plot, "--", color="#b71c1c", linewidth=1.5, label=r"$V_{ocv}$")
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"Voltage (V)")
ax.set_title("② 电压随时间", fontsize=10)
ax.legend()
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[1, 0]
ax.plot(t_plot / 60, I_plot, color="#2e7d32", linewidth=2)
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"Current (A)")
ax.set_title("③ 电流随时间", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[1, 1]
ax.plot(S_plot, Vocv_plot, color="#6a1b9a", linewidth=2)
ax.set_xlabel(r"SOC $S$")
ax.set_ylabel(r"$V_{ocv}$ (V)")
ax.set_title("④ Vocv-SOC 关系", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

fig2.suptitle("图2：ODE 模拟（遗传算法参数）", fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "battery_ode_ga.png"), dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUTPUT_DIR}/battery_ode_ga.png")

plt.show()
