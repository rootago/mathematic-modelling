"""
计算 mobile data 数据集的 RMSE

加载功率模型，对放电段进行 ODE 仿真，计算：
- SOC RMSE（模拟 SOC 与实测 battery_level 的均方根误差，%）
- 功率 RMSE（模型功率与实测 battery_power 的均方根误差，W）
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # code -> 项目根目录
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
# 功率模型路径：优先 aging/mobile/new，其次 output
POWER_MODEL_PATHS = [
    os.path.join(BASE_DIR, "code", "aging", "mobile", "new", "power_model_aging.json"),
    os.path.join(BASE_DIR, "output", "power_model_aging.json"),
]

# ---------- 电池参数 ----------
Q0 = 8.0
R0 = 0.05
R1 = 0.03
C1 = 1000.0
v1 = 0.002
v2 = 0.8
theta1_default = 1e-8
theta2_default = 0.001


def vocv(S, Q=None):
    E0, k, A, B = 3.85, 0.02, 0.25, 3.0
    S_safe = np.clip(S, 1e-6, 1.0)
    Q_val = Q if Q is not None else Q0
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q_val)


def soc_indicator(S):
    S_pct = S * 100
    return 1.0 if (S_pct > 95 or S_pct < 10) else 0.0


def degradation_stress(T, I_current, S, theta_params=None):
    temp_factor = np.exp(0.035 * (T - 25))
    current_factor = 1 + v1 * abs(I_current) + v2 * soc_indicator(S)
    return temp_factor * current_factor


def temperature_degradation_factor(T, theta2):
    return np.exp(-theta2 * T)


def aged_capacity(F, T, theta2, Q_nominal=None):
    Q_val = Q_nominal if Q_nominal is not None else Q0
    y_T = temperature_degradation_factor(T, theta2)
    return Q_val * F * y_T


# ==================== 加载数据 ====================
def load_mobile_data(max_files=None):
    """加载 mobile data 下所有 *_dynamic_processed.csv"""
    pattern = os.path.join(DATA_DIR, "*", "*", "*_dynamic_processed.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到数据文件，路径: {DATA_DIR}")

    if max_files:
        files = files[:max_files]

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["source_file"] = f
            all_dfs.append(df)
        except Exception as e:
            print(f"  跳过 {f}: {e}")

    data = pd.concat(all_dfs, ignore_index=True)
    print(f"共加载 {len(files)} 个文件，{len(data)} 行数据")
    return data


def find_discharge_segments(data, min_duration_sec=1800, min_soc_drop=10):
    """找连续放电段"""
    segments = []
    for source, group in data.groupby("source_file"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        is_discharge = (group["battery_charging_status"] == 3).values

        seg_start = None
        for i in range(len(group)):
            if is_discharge[i]:
                if seg_start is None:
                    seg_start = i
            else:
                if seg_start is not None:
                    seg_end = i
                    seg_df = group.iloc[seg_start:seg_end]
                    duration = (
                        seg_df["timestamp"].iloc[-1] - seg_df["timestamp"].iloc[0]
                    ).total_seconds()
                    soc_drop = (
                        seg_df["battery_level"].iloc[0]
                        - seg_df["battery_level"].iloc[-1]
                    )

                    if duration >= min_duration_sec and soc_drop >= min_soc_drop:
                        segments.append(
                            {
                                "data": seg_df.copy(),
                                "source": source,
                                "duration_min": duration / 60,
                                "soc_start": seg_df["battery_level"].iloc[0],
                                "soc_end": seg_df["battery_level"].iloc[-1],
                                "soc_drop": soc_drop,
                            }
                        )
                    seg_start = None

        if seg_start is not None:
            seg_df = group.iloc[seg_start:]
            duration = (
                seg_df["timestamp"].iloc[-1] - seg_df["timestamp"].iloc[0]
            ).total_seconds()
            soc_drop = (
                seg_df["battery_level"].iloc[0] - seg_df["battery_level"].iloc[-1]
            )
            if duration >= min_duration_sec and soc_drop >= min_soc_drop:
                segments.append(
                    {
                        "data": seg_df.copy(),
                        "source": source,
                        "duration_min": duration / 60,
                        "soc_start": seg_df["battery_level"].iloc[0],
                        "soc_end": seg_df["battery_level"].iloc[-1],
                        "soc_drop": soc_drop,
                    }
                )
    return segments


# ==================== 功率模型 ====================
def load_power_model(json_path=None):
    """从 JSON 加载功率模型"""
    if json_path and os.path.isfile(json_path):
        path = json_path
    else:
        path = None
        for p in POWER_MODEL_PATHS:
            if os.path.isfile(p):
                path = p
                break
    if not path:
        raise FileNotFoundError("未找到 power_model_aging.json")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    coef = cfg["coefficients"]
    params = [
        coef["C1_screen"],
        coef["C2_bright"],
        coef["C3_cpu_usage_freq"],
        coef["C4_throughput"],
        coef["C5_wifi"],
        coef["C6_gps"],
        coef["C7_gps_quad"],
    ]
    intercept = coef.get("Intercept_base", 0.0)
    lam = cfg.get("lambda", 0.03)

    class PowerModel:
        def calc_power_from_row(self, row):
            screen_on = 1.0 if row.get("screen_status", 0) == 1 else 0.0
            bright_norm = min(max(row.get("bright_level", 0), 0), 255) / 255.0

            cpu_usage = min(max(row.get("cpu_usage", 0), 0), 100) / 100.0
            freq_cols = [c for c in row.index if c.startswith("frequency_core")]
            freq_norm = (
                sum(row.get(c, 0) for c in freq_cols) / (3000.0 * max(len(freq_cols), 1))
                if freq_cols
                else 0.0
            )
            cpu_usage_freq = cpu_usage * freq_norm

            throughput = sum(
                row.get(c, 0) for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"]
            ) / 1e6
            throughput = min(max(throughput, 0), 10)

            wifi_on = 1.0 if row.get("wifi_status", 0) == 1 else 0.0
            wifi_int = min(max(row.get("wifi_intensity", -50), -100), -20)
            wifi_exp = wifi_on * np.exp(-lam * wifi_int)

            gps_status = row.get("gps_status", 0)
            gps_on = 1.0 if gps_status > 0 else 0.0
            gps_act_norm = min(max(gps_status, 0), 3) / 3.0 if gps_status > 0 else 0.0
            gps_quad = gps_on * (gps_act_norm ** 2)

            features = [
                screen_on,
                screen_on * bright_norm,
                cpu_usage_freq,
                throughput,
                wifi_exp,
                gps_on,
                gps_quad,
            ]
            P = intercept + sum(p * f for p, f in zip(params, features))
            return max(P, 0.1)

    pm = PowerModel()
    pm.params = params
    pm.intercept = intercept
    pm.lam = lam
    return pm


# ==================== ODE 验证 ====================
def validate_segment(segment, power_model, Q_est=Q0, theta1=theta1_default, theta2=theta2_default):
    """
    用放电段数据驱动 ODE，返回 SOC RMSE（%）和功率 RMSE（W）
    """
    df = segment["data"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds()
    df["P_model"] = df.apply(power_model.calc_power_from_row, axis=1)
    df["P_measured"] = df["battery_power"].abs()

    if "battery_temperature" in df.columns:
        df["T_celsius"] = df["battery_temperature"].fillna(25.0)
    else:
        df["T_celsius"] = 25.0

    t_data = df["t_sec"].values
    P_data = df["P_model"].values
    T_data = df["T_celsius"].values
    SOC_measured = df["battery_level"].values / 100.0

    P_interp = interp1d(
        t_data, P_data, kind="linear", fill_value="extrapolate"
    )
    T_interp = interp1d(
        t_data, T_data, kind="linear", fill_value="extrapolate"
    )

    S0 = SOC_measured[0]
    Up0 = 0.0
    F0 = 1.0
    t_span = (t_data[0], t_data[-1])

    def odefun(t, y):
        S_val = np.clip(y[0], 1e-6, 1.0)
        Up_val = y[1]
        F_val = np.clip(y[2], 0.5, 1.0)
        if S_val <= 1e-6:
            return np.array([0.0, 0.0, 0.0])

        P = float(P_interp(t))
        T = float(T_interp(t))
        Qm = aged_capacity(F_val, T, theta2, Q_est)
        Qm = max(Qm, Q_est * 0.5)

        Vocv_val = vocv(S_val, Qm)
        V_eff = Vocv_val - Up_val
        discriminant = V_eff ** 2 - 4 * R0 * P
        I_val = (
            (V_eff - np.sqrt(discriminant)) / (2 * R0)
            if discriminant >= 0
            else V_eff / (2 * R0)
        )
        I_val = max(I_val, 0.0)
        D = degradation_stress(T, I_val, S_val)

        dS_dt = -I_val / (Qm * 3600.0)
        dUp_dt = -Up_val / (R1 * C1) + I_val / C1
        dF_dt = -theta1 * D
        return np.array([dS_dt, dUp_dt, dF_dt])

    dt_median = np.median(np.diff(t_data)) if len(t_data) > 1 else 60
    max_step = min(max(dt_median * 2, 10), 60)

    sol = solve_ivp(
        odefun, t_span, [S0, Up0, F0],
        method="RK45", dense_output=True, max_step=max_step
    )
    sol_values = sol.sol(t_data)
    SOC_simulated = np.clip(sol_values[0], 0, 1)

    # 返回误差平方列表，便于合并后计算整体 RMSE
    soc_errors_sq = ((SOC_simulated - SOC_measured) * 100) ** 2  # 转为 %
    power_errors_sq = (df["P_model"].values - df["P_measured"].values) ** 2

    return {
        "soc_errors_sq": soc_errors_sq.tolist(),
        "power_errors_sq": power_errors_sq.tolist(),
    }


# ==================== 主程序 ====================
def main(max_files=None):
    print("=" * 60)
    print("计算 mobile data 数据集 RMSE")
    print("=" * 60)

    # 加载数据
    print("\n加载 mobile data...")
    data = load_mobile_data(max_files=max_files)

    # 加载功率模型
    print("加载功率模型...")
    power_model = load_power_model()

    # 找放电段
    print("查找放电段 (时长 ≥ 30min, SOC 下降 ≥ 10%)...")
    segments = find_discharge_segments(data)
    print(f"找到 {len(segments)} 个放电段")

    if not segments:
        print("没有符合条件的放电段")
        return

    # 逐段计算误差
    print(f"\n开始计算 {len(segments)} 个放电段...")
    results = []
    for i, seg in enumerate(segments):
        try:
            print(f"  [{i+1}/{len(segments)}] 处理中... (SOC: {seg['soc_start']:.0f}%->{seg['soc_end']:.0f}%, {seg['duration_min']:.0f}min)", end="")
            r = validate_segment(seg, power_model)
            results.append(r)
            print(f" ✓")
        except Exception as e:
            print(f" ✗ 失败: {e}")

    if not results:
        print("没有成功计算的放电段")
        return

    # 合并所有段的误差，计算整体 RMSE
    all_soc_errors_sq = []
    all_power_errors_sq = []
    for r in results:
        all_soc_errors_sq.extend(r["soc_errors_sq"])
        all_power_errors_sq.extend(r["power_errors_sq"])

    soc_rmse = np.sqrt(np.mean(all_soc_errors_sq))
    power_rmse = np.sqrt(np.mean(all_power_errors_sq))

    print("\n" + "=" * 60)
    print("RMSE 结果")
    print("=" * 60)
    print(f"  SOC RMSE:  {soc_rmse:.4f} %")
    print(f"  功率 RMSE: {power_rmse:.4f} W")
    print(f"\n共 {len(results)} 个放电段，{len(all_soc_errors_sq)} 个数据点")

    # 计算误差（非平方）
    soc_errors = np.sqrt(all_soc_errors_sq)  # 绝对误差 (%)
    power_errors = np.sqrt(all_power_errors_sq)  # 绝对误差 (W)

    # ==================== 可视化 ====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SOC 误差直方图
    ax1 = axes[0]
    ax1.hist(soc_errors, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(soc_rmse, color="red", linestyle="--", linewidth=2, label=f"RMSE = {soc_rmse:.2f}%")
    ax1.set_xlabel("SOC 绝对误差 (%)")
    ax1.set_ylabel("频数")
    ax1.set_title("SOC 误差分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 功率误差直方图
    ax2 = axes[1]
    ax2.hist(power_errors, bins=50, color="coral", edgecolor="white", alpha=0.85)
    ax2.axvline(power_rmse, color="red", linestyle="--", linewidth=2, label=f"RMSE = {power_rmse:.2f}W")
    ax2.set_xlabel("功率绝对误差 (W)")
    ax2.set_ylabel("频数")
    ax2.set_title("功率误差分布")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(SCRIPT_DIR, "rmse_histogram.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"直方图已保存: {fig_path}")

    # 保存结果
    out_path = os.path.join(SCRIPT_DIR, "rmse_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "soc_rmse_pct": float(soc_rmse),
                "power_rmse_w": float(power_rmse),
                "n_segments": len(results),
                "n_points": len(all_soc_errors_sq),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    import sys
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_files=max_files)
