"""
2026 MCM Problem A：智能手机电池消耗建模 — 仅实现赛题要求1（连续时间模型）

赛题要求1：使用连续时间方程表示荷电状态（SOC）；数据作为支持（验证），非替代；非纯离散拟合/黑箱。

模型假设：锂聚合物（Li-Po）电池，Tremblay 模型。等效电路 + OCV-SOC 经验关系。
控制方程（连续时间）：
  dS/dt = -I/Q0，I = P_total/U，U = Vocv - I*r，
  Vocv(S) = E0 - K/S + A*exp(-B*Qnom*(1-S))。（仅与 SOC S 有关）

========== 参数说明 ==========
- 采用典型锂聚合物（3.7V/4.2V 体系）智能手机用标准参数，直接固定，不做最小二乘拟合。
- 图1：U 实测 vs 模型、残差、数据 SOC（若有数据）；图2：ODE 模拟 S(t), U(t), I(t), Vocv(S)。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 中文字体，避免 "Glyph missing" 警告（Windows 常用 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 常数（Tremblay 标准参数，Li-Po 3.7V/4.2V 智能手机用，直接固定）----------
# 参考：Tremblay 模型，公式 Vocv(S) = E0 - K/S + A*exp(-B*Qnom*(1-S))
Ptotal = 20.0
Q0 = 3.0          # Qnom 标称容量 3.0 Ah (3000 mAh)
r = 0.022         # 内阻 (Ω)
E0 = 3.85         # 电池恒定电压 (V)，比 3.7V 高，因需减去后面项
k = 0.02          # K 极化电压幅度 (V)，控制电量耗尽时电压下降快慢
A = 0.25          # 指数区幅度 (V)，控制刚充满时的虚高电压
B = 3.0           # 指数区常数 (Ah)^-1，控制虚高电压衰减快慢

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


# ========== 1. 固定参数（无拟合）；若有工况数据则加载用于图1 实测 vs 模型 ==========
print("使用固定 Tremblay 标准参数: E0=%.2f V, K=%.2f V, A=%.2f V, B=%.1f (Ah)^-1, Qnom=%.1f Ah"
      % (E0, k, A, B, Q0))
with open(os.path.join(OUTPUT_DIR, "vocv_fit_params.json"), "w", encoding="utf-8") as f:
    json.dump({
        "method": "Fixed (Li-Po standard, no fitting)",
        "E0": E0, "k": k, "A": A, "B": B, "Q0": Q0, "r": r
    }, f, indent=2)

t_fit = I_fit = U_obs_fit = S_fit = None
if os.path.exists(DYNAMIC_DATA_PATH):
    t_fit, I_fit, U_obs_fit, S_fit = load_dynamic_data(DYNAMIC_DATA_PATH)
    if t_fit is not None:
        print("使用的列: Test_Time(s), Current(A), Voltage(V), Discharge_Capacity(Ah); "
              "SOC = 1 - Discharge_Capacity(Ah)/Q0，范围 [0, 1]。")
        print("仅使用放电阶段（Arbin 惯例：Current(A) > 0 为放电，已过滤 I > 0.01 A）。")
        print(f"SOC 范围: 首 {S_fit[0]:.4f} ~ 末 {S_fit[-1]:.4f}")
        step = max(1, len(t_fit) // 2000)
        t_fit, I_fit = t_fit[::step], I_fit[::step]
        U_obs_fit, S_fit = U_obs_fit[::step], S_fit[::step]
        # 用固定参数计算模型电压，得到残差（仅用于图示与参考）
        U_model_fit = vocv(S_fit, E0, k, A, B, Q0) - I_fit * r
        rmse = np.sqrt(np.mean((U_obs_fit - U_model_fit) ** 2))
        print(f"固定参数下与实测对比: RMSE = {rmse:.6f} V")
else:
    print("未找到工况数据:", DYNAMIC_DATA_PATH, "（跳过图1 实测对比）")

# ========== 2. ODE：dS/dt = -I/Q0，用固定的 k,A,B；S 限制在 [0, 1] ==========
# 注意单位：t 为秒，Q0 为 Ah，I 为 A；dS/dt(每h)=-I/Q0 => dS/dt(每秒)=-I/(Q0*3600)
def odefun(t, S):
    S_val = np.clip(S[0], 1e-6, 1.0)  # SOC 不超出 [0, 1]
    if S_val <= 1e-6:
        return np.array([0.0])  # 放空后不再下降
    Vocv_val = vocv(S_val, E0, k, A, B, Q0)
    U_val = u_from_vocv(Vocv_val, Ptotal, r)
    I_val = Ptotal / (U_val + 1e-12)
    return np.array([-I_val / (Q0 * 3600.0)])  # Q0 为 Ah，t 为 s，故除以 3600

# 与 DST_50SOC 一致：起始 SOC=50%（Arbin 命名约定）
S0 = 0.5
t_span = (0, 3600)
sol = solve_ivp(odefun, t_span, [S0], method="RK45", dense_output=True)
t_plot = np.linspace(0, t_span[1], 300)
S_plot = sol.sol(t_plot)[0]
Vocv_plot = vocv(S_plot, E0, k, A, B, Q0)
U_plot = u_from_vocv(Vocv_plot, Ptotal, r)
I_plot = Ptotal / U_plot

# ========== 3. 绘图 ==========
# ---------- 图1：fixed1.png（全部来自「实测数据」DST 放电段）----------
# 目的：看用拟合的 k,A,B 算出的 U 模型 与 实测 U 是否一致；以及数据里的 SOC 随时间怎么变
if t_fit is not None:
    Vocv_fit = vocv(S_fit, E0, k, A, B, Q0)
    U_model_fit = Vocv_fit - I_fit * r
    t_rel_min = (t_fit - t_fit.min()) / 60.0  # 相对时间，分钟，从 0 开始
    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes1[0].plot(t_rel_min, U_obs_fit, alpha=0.7, label="U 实测", color="#1565c0")
    axes1[0].plot(t_rel_min, U_model_fit, alpha=0.9, label="U 模型 (Vocv(S)-Ir)", color="#b71c1c", linewidth=1.2)
    axes1[0].set_ylabel(r"Voltage (V)")
    axes1[0].set_xlim(0, None)
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.5)
    axes1[0].set_title("① 端电压：实测 vs 模型（固定参数，非拟合）", fontsize=10)
    axes1[1].plot(t_rel_min, U_obs_fit - U_model_fit, color="#2e7d32", alpha=0.8)
    axes1[1].set_ylabel(r"$U_{obs} - U_{model}$")
    axes1[1].set_xlim(0, None)
    axes1[1].grid(True, alpha=0.5)
    axes1[1].set_title("② 模型残差：越接近 0 说明模型越准", fontsize=10)
    axes1[2].plot(t_rel_min, S_fit, color="#6a1b9a", linewidth=1.5, label=r"数据 SOC ($1 - D_{ch}/Q_0$)")
    axes1[2].set_ylabel(r"SOC")
    axes1[2].set_ylim(0, 1.05)
    axes1[2].set_xlabel(r"Time (min), elapsed from start")
    axes1[2].set_xlim(0, None)
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.5)
    axes1[2].set_title("③ 数据中的 SOC 随时间变化（由放电容量算得，仅放电段）", fontsize=10)
    fig1.suptitle("图1：固定参数模型 vs 实测（数据来源：data2 DST 放电段）", fontsize=12)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "fixed1.png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved: {OUTPUT_DIR}/fixed1.png")

# ---------- 图2：fixed2.png（全部来自「数值模拟」，不是实测）----------
# 目的：用固定参数，假设「恒功率 20W 放电 1 小时」，看 S、U、I、Vocv 随时间/随 S 的变化
S_plot = np.clip(S_plot, 0.0, 1.0)  # 绘图时保证 SOC 在 [0, 1]
fig2, axes2 = plt.subplots(2, 2, figsize=(9, 7))
ax = axes2[0, 0]
ax.plot(t_plot / 60, S_plot, color="#0d47a1", linewidth=2)
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"SOC $S$")
ax.set_ylim(0, 1.05)
ax.set_title("① 模拟的 SOC 随时间（从 50% 起恒功率放电，约 10 min 放空）", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[0, 1]
ax.plot(t_plot / 60, U_plot, color="#1565c0", linewidth=2, label=r"$U$")
ax.plot(t_plot / 60, Vocv_plot, "--", color="#b71c1c", linewidth=1.5, label=r"$V_{ocv}$")
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"Voltage (V)")
ax.set_title("② 模拟的端电压 U 与开路电压 Vocv 随时间", fontsize=10)
ax.legend()
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[1, 0]
ax.plot(t_plot / 60, I_plot, color="#2e7d32", linewidth=2)
ax.set_xlabel(r"Time (min)")
ax.set_ylabel(r"Current (A)")
ax.set_title("③ 模拟的电流 I = P/U（恒功率故 U 降则 I 升）", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

ax = axes2[1, 1]
ax.plot(S_plot, Vocv_plot, color="#6a1b9a", linewidth=2)
ax.set_xlabel(r"State of charge $S$")
ax.set_ylabel(r"$V_{ocv}$ (V)")
ax.set_title("④ 开路电压 Vocv 与 SOC 的关系（Tremblay 固定参数曲线）", fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor("#fafafa")

fig2.suptitle("图2：ODE 数值模拟（起始 SOC=50%，恒功率 20W 放电 1h）", fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fixed2.png"), dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUTPUT_DIR}/fixed2.png")

# plt.show() 会阻塞直到关闭图形窗口；若只需保存图片不需弹窗，可注释掉
# plt.show()
plt.close("all")
print("Done. 图片已保存至 output/，未弹窗。")
