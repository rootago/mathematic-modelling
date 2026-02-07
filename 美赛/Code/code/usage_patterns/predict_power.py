"""
多场景手机耗电预测

根据活动类型占比与参数映射表，结合 mobile data 实际数据统计，
用功率模型预测各场景功耗与续航。

支持场景:
- 上班族: 28.1%导航, 44.8%待机, 15.8%视频, 11.3%游戏
- 地铁: 53.7%视频, 56.3%待机 (31.5+10.5+11.7 视频, 14.7+16.8+24.8 待机, 归一化)
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
POWER_MODEL_PATHS = [
    os.path.join(BASE_DIR, "code", "aging", "mobile", "new", "power_model_aging.json"),
    os.path.join(BASE_DIR, "output", "power_model_aging.json"),
]

Q0 = 8.0  # Ah
V_avg = 3.9  # V

# ---------- 活动参数映射表（来自题目）------------
# 列：屏幕亮度、CPU负载、网络活动、GPS状态（均归一化 0~1，GPS 为 0/1）
ACTIVITY_TABLE = {
    "待机": {"bright": [0, 0.1], "cpu": [0, 0.1], "network": [0, 0.15], "gps": 0},
    "打电话": {"bright": [0.2, 0.45], "cpu": [0.15, 0.3], "network": [0.25, 0.45], "gps": 0},
    "看视频": {"bright": [0.55, 0.85], "cpu": [0.45, 0.75], "network": [0.1, 0.85], "gps": 0},
    "玩游戏": {"bright": [0.75, 1], "cpu": [0.75, 1], "network": [0.2, 0.65], "gps": 0},
    "开导航": {"bright": [0.65, 0.95], "cpu": [0.55, 0.85], "network": [0.65, 0.95], "gps": 1},
}

# ---------- 使用模式占比 ----------
# 各模式为 (活动名, 占比) 列表，同类活动占比会合并
USAGE_PATTERNS = {
    "上班族": [
        ("开导航", 0.281),
        ("待机", 0.196),
        ("看视频", 0.158),
        ("待机", 0.252),
        ("玩游戏", 0.113),
    ],
    "地铁": [
        ("看视频", 0.315),
        ("待机", 0.147),
        ("看视频", 0.105),
        ("待机", 0.168),
        ("待机", 0.248),
        ("看视频", 0.117),
    ],
    "宅家": [
        ("玩游戏", 0.25),
        ("看视频", 0.105),
        ("玩游戏", 0.25),
        ("待机", 0.035),
        ("看视频", 0.36),
    ],
    "微商": [
        ("看视频", 0.21),
        ("待机", 0.021),
        ("看视频", 0.21),
        ("开导航", 0.114),
        ("打电话", 0.336),
        ("待机", 0.021),
    ],
    "持续游戏": [("玩游戏", 1.0)],  # 一直玩游戏直到电量耗尽
}

# Figure labels: Chinese -> English (for chart output only)
NAME_EN = {"上班族": "Office Worker", "地铁": "Commute", "宅家": "Home", "微商": "WeChat Biz", "持续游戏": "Gaming Only"}
ACT_EN = {"待机": "Standby", "打电话": "Call", "看视频": "Video", "玩游戏": "Gaming", "开导航": "Navigation"}
# 堆叠柱状图：按功率成分固定顺序与配色（参考蓝/绿/橙，全彩色无米灰）
POWER_COMPONENT_ORDER = ["Base", "Screen", "CPU", "Network", "GPS"]
POWER_COMPONENT_LABELS = {"Base": "Base", "Screen": "Screen", "CPU": "CPU", "Network": "Network", "GPS": "GPS"}
POWER_COMPONENT_COLORS = {"Base": "#00acc1", "Screen": "#1976d2", "CPU": "#f57c00", "Network": "#43a047", "GPS": "#7b1fa2"}
# 续航柱状图（清晰蓝）
DURATION_BAR_COLOR = "#1e88e5"
# 柱状图顶部留白比例（避免柱子顶到图边缘）
BAR_TOP_MARGIN = 1.18
# 老化模型：假设每年完整放电次数（用于 Qm/Q0）
CYCLES_PER_YEAR = 365


def load_mobile_data_stats(max_folders=20, nrows_per_file=5000):
    """
    从 mobile data 加载实际数据，统计各参数的分布范围。
    与 mobile_new.py 用同一套文件选择逻辑：按日期文件夹排序，取前 max_folders 个，
    每个日期下的所有 *_dynamic_processed.csv 都加载。
    """
    folders = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()]
    )[:max_folders]

    dfs = []
    for folder in folders:
        pattern = os.path.join(DATA_DIR, folder, "*", "*_dynamic_processed.csv")
        files = glob.glob(pattern)
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False, nrows=nrows_per_file)
                dfs.append(df)
            except Exception:
                continue

    if not dfs:
        return None
    data = pd.concat(dfs, ignore_index=True)

    # 只保留 screen_status=1 的亮屏数据（更贴近真实使用）
    if "screen_status" in data.columns:
        mask = data["screen_status"] == 1
        if mask.sum() >= 100:
            data = data[mask]

    stats = {}

    # 屏幕亮度 0~255
    if "bright_level" in data.columns:
        br = pd.to_numeric(data["bright_level"], errors="coerce").dropna()
        stats["bright_min"] = br.quantile(0.05)
        stats["bright_max"] = br.quantile(0.95)

    # CPU 使用率 0~100
    if "cpu_usage" in data.columns:
        cu = pd.to_numeric(data["cpu_usage"], errors="coerce").dropna()
        stats["cpu_min"] = cu.quantile(0.05)
        stats["cpu_max"] = cu.quantile(0.95)

    # 频率（取 core 均值，MHz）
    freq_cols = [c for c in data.columns if c.startswith("frequency_core")]
    if freq_cols:
        freq = data[freq_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        stats["freq_min"] = freq.quantile(0.05)
        stats["freq_max"] = freq.quantile(0.95)

    # 吞吐量 (wifi_rx + wifi_tx + mobile_rx + mobile_tx) / 1e6 -> MB/s
    tp_cols = [c for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"] if c in data.columns]
    if tp_cols:
        tp = data[tp_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) / 1e6
        tp = tp.clip(0, 10)
        stats["throughput_min"] = tp.quantile(0.05)
        stats["throughput_max"] = tp.quantile(0.95)

    # WiFi 信号强度 dBm
    if "wifi_intensity" in data.columns:
        wi = pd.to_numeric(data["wifi_intensity"], errors="coerce").dropna()
        wi = wi[(wi >= -100) & (wi <= -20)]
        if len(wi) > 0:
            stats["wifi_int_min"] = wi.quantile(0.05)
            stats["wifi_int_max"] = wi.quantile(0.95)

    return stats


def load_power_model():
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
    aging = cfg.get("aging_parameters", {})
    v1 = float(aging.get("v1", 0.002))
    v2 = float(aging.get("v2", 0.8))
    theta1_raw = aging.get("theta1", 1e-8)
    theta2_raw = aging.get("theta2", 0.001)
    theta1 = float(theta1_raw) if isinstance(theta1_raw, (int, float)) else 1e-8
    theta2 = float(theta2_raw) if isinstance(theta2_raw, (int, float)) else 0.001

    return {
        "params": params,
        "intercept": intercept,
        "lam": lam,
        "v1": v1,
        "v2": v2,
        "theta1": theta1,
        "theta2": theta2,
    }


def table_to_row(activity_name, stats, use_midpoint=True):
    """
    将映射表中的活动参数转换为功率模型所需的 row 格式。
    使用 actual data 统计范围将 [0,1] 映射到实际单位。
    """
    act = ACTIVITY_TABLE[activity_name]
    s = stats or {}

    # 屏幕：bright_level 0~255，screen_status=1
    b_min, b_max = act["bright"]
    b_norm = (b_min + b_max) / 2 if use_midpoint else b_min
    bright_level = int(
        np.clip(
            b_norm * (s.get("bright_max", 255) - s.get("bright_min", 0)) + s.get("bright_min", 0),
            0, 255
        )
    )

    # CPU 0~100
    c_min, c_max = act["cpu"]
    c_norm = (c_min + c_max) / 2 if use_midpoint else c_min
    cpu_usage = np.clip(
        c_norm * (s.get("cpu_max", 100) - s.get("cpu_min", 0)) + s.get("cpu_min", 0),
        0, 100
    )

    # 频率：CPU 负载高时频率也高，用 cpu_norm 线性映射到 freq 范围
    freq_min = s.get("freq_min", 600)
    freq_max = s.get("freq_max", 2400)
    freq_avg = c_norm * (freq_max - freq_min) + freq_min
    # 8 核同频，freq_norm = freq_avg / 3000（模型假设）
    freq_per_core = freq_avg

    # 吞吐量 MB/s，网络 [0,1] 映射到 throughput 范围
    n_min, n_max = act["network"]
    n_norm = (n_min + n_max) / 2 if use_midpoint else n_min
    throughput_mbs = np.clip(
        n_norm * (s.get("throughput_max", 5) - s.get("throughput_min", 0)) + s.get("throughput_min", 0),
        0, 10
    )
    throughput_bytes = throughput_mbs * 1e6
    wifi_rx = throughput_bytes / 2
    wifi_tx = throughput_bytes / 2
    mobile_rx = 0
    mobile_tx = 0

    # WiFi 强度：有网络时用典型值
    wifi_status = 1 if n_norm > 0.1 else 0
    wifi_intensity = s.get("wifi_int_min", -65) if wifi_status else -100

    # GPS
    gps_status = act["gps"]
    if gps_status == 1:
        gps_status = 3

    row = {
        "screen_status": 1,
        "bright_level": bright_level,
        "cpu_usage": cpu_usage,
        "frequency_core0": freq_per_core,
        "frequency_core1": freq_per_core,
        "frequency_core2": freq_per_core,
        "frequency_core3": freq_per_core,
        "frequency_core4": freq_per_core,
        "frequency_core5": freq_per_core,
        "frequency_core6": freq_per_core,
        "frequency_core7": freq_per_core,
        "wifi_rx": wifi_rx,
        "wifi_tx": wifi_tx,
        "mobile_rx": mobile_rx,
        "mobile_tx": mobile_tx,
        "wifi_status": wifi_status,
        "wifi_intensity": wifi_intensity,
        "gps_status": gps_status,
    }
    return row


def calc_power(row, model):
    """功率模型计算"""
    p = model["params"]
    lam = model["lam"]
    intercept = model["intercept"]

    screen_on = 1.0 if row.get("screen_status", 0) == 1 else 0.0
    bright_norm = min(max(row.get("bright_level", 0), 0), 255) / 255.0

    cpu_usage = min(max(row.get("cpu_usage", 0), 0), 100) / 100.0
    freq_cols = [f"frequency_core{i}" for i in range(8)]
    freq_sum = sum(row.get(c, 0) for c in freq_cols)
    freq_norm = freq_sum / (3000.0 * 8) if freq_sum else 0.0
    cpu_usage_freq = cpu_usage * min(freq_norm, 1.0)

    throughput = (
        sum(row.get(c, 0) for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"]) / 1e6
    )
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
    P = intercept + sum(pi * f for pi, f in zip(p, features))
    return max(P, 0.1)


def calc_power_components(row, model):
    """
    按功率成分分解：屏幕、CPU、网络、GPS（与功率模型各项对应）。
    返回 dict：Base, Screen, CPU, Network, GPS，单位 W，均为非负，总和 >= 0.1。
    """
    p = model["params"]
    lam = model["lam"]
    intercept = model["intercept"]

    screen_on = 1.0 if row.get("screen_status", 0) == 1 else 0.0
    bright_norm = min(max(row.get("bright_level", 0), 0), 255) / 255.0
    cpu_usage = min(max(row.get("cpu_usage", 0), 0), 100) / 100.0
    freq_cols = [f"frequency_core{i}" for i in range(8)]
    freq_sum = sum(row.get(c, 0) for c in freq_cols)
    freq_norm = freq_sum / (3000.0 * 8) if freq_sum else 0.0
    cpu_usage_freq = cpu_usage * min(freq_norm, 1.0)
    throughput = (
        sum(row.get(c, 0) for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"]) / 1e6
    )
    throughput = min(max(throughput, 0), 10)
    wifi_on = 1.0 if row.get("wifi_status", 0) == 1 else 0.0
    wifi_int = min(max(row.get("wifi_intensity", -50), -100), -20)
    wifi_exp = wifi_on * np.exp(-lam * wifi_int)
    gps_status = row.get("gps_status", 0)
    gps_on = 1.0 if gps_status > 0 else 0.0
    gps_act_norm = min(max(gps_status, 0), 3) / 3.0 if gps_status > 0 else 0.0
    gps_quad = gps_on * (gps_act_norm ** 2)

    base = max(0.0, intercept)
    screen = max(0.0, p[0] * screen_on + p[1] * screen_on * bright_norm)
    cpu = max(0.0, p[2] * cpu_usage_freq)
    network = max(0.0, p[3] * throughput + p[4] * wifi_exp)
    gps = max(0.0, p[5] * gps_on + p[6] * gps_quad)
    total = base + screen + cpu + network + gps
    if total < 0.1:
        base += 0.1 - total
        total = 0.1
    return {"Base": base, "Screen": screen, "CPU": cpu, "Network": network, "GPS": gps}


# ---------- 老化模型（与 mobile_tmp 一致：Qm = Q0 * F * y(T)，dF/dt = -θ₁*D）----------
def _soc_indicator(S):
    """SOC 指示：S>95% 或 S<10% 时为 1，否则 0。S 为 0~1。"""
    S_pct = S * 100
    return 1.0 if (S_pct > 95 or S_pct < 10) else 0.0


def _degradation_stress(T, I_current, S, v1, v2):
    """退化应力 D = exp(0.035*(T-25)) * (1 + v1*|I| + v2*I(SOC))"""
    temp_factor = np.exp(0.035 * (T - 25))
    current_factor = 1 + v1 * abs(I_current) + v2 * _soc_indicator(S)
    return temp_factor * current_factor


def _temperature_degradation_factor(T, theta2):
    """y(T) = exp(-θ₂ * T)"""
    return np.exp(-theta2 * T)


def _aged_capacity(F, T, theta2, Q_nominal):
    """Qm = Q0 * F * y(T)，在 -20°C 时额外乘以 0.5（低温容量折损）"""
    y_T = _temperature_degradation_factor(T, theta2)
    Qm = Q_nominal * F * y_T
    # 低温下容量额外折损
    if T <= -20:
        Qm *= 0.5
    return Qm


def compute_aging_delta_F_one_cycle(mix_sequence, duration_h, activity_powers, model, T_ambient=25.0):
    """
    对一次完整放电（100%→0%）按老化模型积分 dF/dt = -θ₁*D，返回该周期内 F 的下降量 delta_F。
    D = degradation_stress(T, I, S)；I = P/V_avg，T 为环境温度（可指定），S 为分段内平均 SOC。
    """
    theta1 = model["theta1"]
    v1, v2 = model["v1"], model["v2"]
    energy_total_wh = Q0 * V_avg
    t_list = [0.0]
    soc_list = [100.0]
    soc_current = 100.0
    t_current = 0.0
    delta_F_total = 0.0

    for act, pct in mix_sequence:
        P_act = activity_powers[act]
        seg_duration_h = duration_h * pct
        if seg_duration_h <= 0:
            continue
        soc_drop = 100 * P_act * seg_duration_h / energy_total_wh
        soc_next = max(0, soc_current - soc_drop)
        S_mid_pct = (soc_current + soc_next) / 2.0
        S_mid = S_mid_pct / 100.0
        I_avg = P_act / V_avg
        D = _degradation_stress(T_ambient, I_avg, S_mid, v1, v2)
        delta_F_total += theta1 * D * seg_duration_h
        t_current += seg_duration_h
        soc_current = soc_next
        if soc_current <= 0:
            break

    return delta_F_total


def compute_qm_over_q0_per_pattern(all_results, activity_powers, model, cycles_per_year=365, T_ambient=25.0):
    """
    对每种行为模式：假设每年 cycles_per_year 次完整放电，积分老化得 F(1年)，
    Qm/Q0 = F * y(T)。T 为环境温度（可指定）。
    在 -20°C 时，额外乘以 0.5（低温容量折损）。
    返回 dict: pattern_name -> Qm/Q0
    """
    theta2 = model["theta2"]
    y_T = _temperature_degradation_factor(T_ambient, theta2)
    out = {}
    for name, res in all_results.items():
        mix_sequence = mix_to_sequence(USAGE_PATTERNS[name])
        duration_h = res["duration_h"]
        delta_F_one = compute_aging_delta_F_one_cycle(mix_sequence, duration_h, activity_powers, model, T_ambient)
        F_after = 1.0 - cycles_per_year * delta_F_one
        F_after = np.clip(F_after, 0.5, 1.0)
        qm_q0 = F_after * y_T
        # 低温下容量额外折损
        if T_ambient <= -20:
            qm_q0 *= 0.5
        out[name] = float(np.clip(qm_q0, 0.0, 1.0))
    return out


def mix_to_dict(mix_list):
    """合并同类活动占比，并归一化使总和为 1"""
    d = {}
    for act, pct in mix_list:
        d[act] = d.get(act, 0) + pct
    total = sum(d.values())
    if total > 0:
        d = {k: v / total for k, v in d.items()}
    return d


def mix_to_sequence(mix_list):
    """保留活动先后顺序，归一化，用于 SOC 分段模拟"""
    total = sum(pct for _, pct in mix_list)
    if total <= 0:
        return []
    return [(act, pct / total) for act, pct in mix_list]


def run_one_pattern(name, mix_dict, model, stats, activity_powers):
    """对单个模式计算功率与续航"""
    P_avg = sum(activity_powers[act] * pct for act, pct in mix_dict.items())
    energy_wh = Q0 * V_avg
    duration_h = energy_wh / P_avg
    return {
        "mix": mix_dict,
        "avg_power_w": P_avg,
        "duration_h": duration_h,
    }


def compute_soc_curve(mix_sequence, activity_powers, duration_h, Q_effective=None):
    """
    计算单条 SOC 预测曲线数据 (t_arr, soc_arr)。
    给定每时段活动占比与续航时长，按各活动耗电功率模拟 SOC 随时间变化至耗尽。
    Q_effective: 有效容量（Ah），如果为 None 则使用 Q0。
    """
    t_list = [0.0]
    soc_list = [100.0]
    soc_current = 100.0
    t_current = 0.0
    Q_used = Q_effective if Q_effective is not None else Q0
    energy_total_wh = Q_used * V_avg

    for act, pct in mix_sequence:
        P_act = activity_powers[act]
        seg_duration_h = duration_h * pct
        if seg_duration_h <= 0:
            continue
        soc_drop = 100 * P_act * seg_duration_h / energy_total_wh
        soc_next = max(0, soc_current - soc_drop)
        t_next = t_current + seg_duration_h

        n_pts = max(2, int(seg_duration_h * 20))
        t_seg = np.linspace(t_current, t_next, n_pts)
        soc_seg = np.linspace(soc_current, soc_next, n_pts)

        t_list.extend(t_seg[1:].tolist())
        soc_list.extend(soc_seg[1:].tolist())
        soc_current = soc_next
        t_current = t_next
        if soc_current <= 0:
            break

    return np.array(t_list), np.array(soc_list)


def plot_all_soc_curves_one_figure(all_results, activity_powers, out_path, T_ambient=25.0, qm_q0_dict=None):
    """
    把所有使用模式的 SOC 预测曲线画到一张图上，每条线一个模式。
    T_ambient: 环境温度
    qm_q0_dict: 各模式在该温度下的 Qm/Q0 比值字典（用于调整有效容量）
    """
    LINE_COLORS = ["#0d9488", "#0891b2", "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"]

    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        ax.set_facecolor("white")

        t_global_max = 0.0
        for i, (name, res) in enumerate(all_results.items()):
            mix_sequence = mix_to_sequence(USAGE_PATTERNS[name])
            duration_h = res["duration_h"]
            
            # 根据温度调整有效容量
            Q_effective = Q0
            if qm_q0_dict and name in qm_q0_dict:
                Q_effective = Q0 * qm_q0_dict[name]
            
            t_arr, soc_arr = compute_soc_curve(mix_sequence, activity_powers, duration_h, Q_effective)
            t_global_max = max(t_global_max, t_arr[-1] if len(t_arr) else 0)
            name_en = NAME_EN.get(name, name)
            color = LINE_COLORS[i % len(LINE_COLORS)]
            ax.plot(t_arr, soc_arr, color=color, linewidth=2.2, label=f"{name_en} ({duration_h:.1f}h)", zorder=3)

        ax.set_xlabel("Time (h)", fontsize=16, color="black")
        ax.set_ylabel("SOC (%)", fontsize=16, color="black")
        title_temp = f"SOC Prediction by Usage Pattern ({T_ambient}°C)"
        ax.set_title(title_temp, fontsize=18, color="black", pad=12)
        ax.set_xlim(0, t_global_max * 1.02 if t_global_max > 0 else 1)
        ax.set_ylim(0, 105)
        ax.tick_params(colors="black", labelsize=14)
        ax.set_axisbelow(False)
        ax.legend(loc="upper right", framealpha=0.9, facecolor="white", edgecolor="#e5e7eb", fontsize=13)
        ax.grid(True, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()


def plot_all_soc_curves_three_temps_one_figure(all_results, activity_powers, qm_q0_by_temp, out_path):
    """
    把三个温度（-20°C, 25°C, 45°C）的 SOC 预测曲线画到一张图上，从上到下排列。
    比例参考：三行一列，整体竖长，每子图横向略宽。
    """
    LINE_COLORS = ["#0d9488", "#0891b2", "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"]
    temperatures = [-20, 25, 45]

    # 先计算三个温度下的最大时间，使 x 轴统一
    t_global_max = 0.0
    for T in temperatures:
        qm_q0_dict = qm_q0_by_temp[T]
        for name, res in all_results.items():
            mix_sequence = mix_to_sequence(USAGE_PATTERNS[name])
            duration_h = res["duration_h"]
            Q_effective = Q0 * qm_q0_dict.get(name, 1.0)
            t_arr, _ = compute_soc_curve(mix_sequence, activity_powers, duration_h, Q_effective)
            if len(t_arr) > 0:
                t_global_max = max(t_global_max, t_arr[-1])
    x_max = t_global_max * 1.02 if t_global_max > 0 else 1.0

    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), facecolor="white", sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for idx, T in enumerate(temperatures):
            ax = axes[idx]
            ax.set_facecolor("white")
            qm_q0_dict = qm_q0_by_temp[T]

            for i, (name, res) in enumerate(all_results.items()):
                mix_sequence = mix_to_sequence(USAGE_PATTERNS[name])
                duration_h = res["duration_h"]
                Q_effective = Q0 * qm_q0_dict.get(name, 1.0)
                t_arr, soc_arr = compute_soc_curve(mix_sequence, activity_powers, duration_h, Q_effective)
                name_en = NAME_EN.get(name, name)
                color = LINE_COLORS[i % len(LINE_COLORS)]
                ax.plot(t_arr, soc_arr, color=color, linewidth=2.2, label=f"{name_en} ({duration_h:.1f}h)", zorder=3)

            ax.set_ylabel("SOC (%)", fontsize=16, color="black")
            ax.set_title(f"SOC Prediction by Usage Pattern ({T}°C)", fontsize=18, color="black", pad=12)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 105)
            ax.tick_params(colors="black", labelsize=14)
            ax.set_axisbelow(False)
            ax.legend(loc="upper right", framealpha=0.9, facecolor="white", edgecolor="#e5e7eb", fontsize=13)
            ax.grid(True, alpha=0.35)
            for spine in ax.spines.values():
                spine.set_color("#d1d5db")

        axes[-1].set_xlabel("Time (h)", fontsize=16, color="black")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.28)
        plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()


def main():
    print("=" * 60)
    print("多场景手机耗电预测")
    print("=" * 60)

    print("\n加载功率模型...")
    model = load_power_model()

    print("加载 mobile data 统计实际参数范围...")
    stats = load_mobile_data_stats()
    if stats:
        print(f"  亮度: {stats.get('bright_min', 0):.0f}~{stats.get('bright_max', 255):.0f}")
        print(f"  CPU: {stats.get('cpu_min', 0):.0f}~{stats.get('cpu_max', 100):.0f}%")
        print(f"  频率: {stats.get('freq_min', 600):.0f}~{stats.get('freq_max', 2400):.0f} MHz")
    else:
        print("  使用默认范围")

    # 各活动典型功率（所有模式共用）
    all_acts = set()
    for mix_list in USAGE_PATTERNS.values():
        for act, _ in mix_list:
            all_acts.add(act)
    activity_powers = {}
    activity_components = {}
    for act in all_acts:
        row = table_to_row(act, stats)
        activity_powers[act] = calc_power(row, model)
        activity_components[act] = calc_power_components(row, model)
    print("\n各活动典型功率 (W):")
    for act, P in activity_powers.items():
        print(f"  {act}: {P:.3f} W")

    # 逐模式计算
    all_results = {}
    for name, mix_list in USAGE_PATTERNS.items():
        mix_dict = mix_to_dict(mix_list)
        res = run_one_pattern(name, mix_dict, model, stats, activity_powers)
        all_results[name] = res

        print(f"\n{'='*40}")
        print(f"【{name}】")
        print("活动占比:")
        for act, pct in mix_dict.items():
            print(f"  {act}: {pct*100:.1f}%")
        print(f"加权平均功率: {res['avg_power_w']:.3f} W")
        print(f"续航预估: {res['duration_h']:.1f} h ({res['duration_h']/24:.1f} 天)")

    # 计算三个温度下的老化比例（用于 SOC 曲线）
    temperatures = [-20, 25, 45]
    qm_q0_by_temp = {}
    for T in temperatures:
        qm_q0_by_temp[T] = compute_qm_over_q0_per_pattern(
            all_results, activity_powers, model, cycles_per_year=CYCLES_PER_YEAR, T_ambient=T
        )
    
    # 三张 SOC 图合并为一张（从上到下：-20°C, 25°C, 45°C）
    out_combined = os.path.join(SCRIPT_DIR, "soc_all_patterns_combined.png")
    plot_all_soc_curves_three_temps_one_figure(all_results, activity_powers, qm_q0_by_temp, out_combined)
    print(f"三温度 SOC 合并图已保存: {out_combined}")
    
    # 为每个温度单独生成 SOC 曲线图（可选）
    for T in temperatures:
        out_path = os.path.join(SCRIPT_DIR, f"soc_all_patterns_{T}C.png")
        plot_all_soc_curves_one_figure(all_results, activity_powers, out_path, T_ambient=T, qm_q0_dict=qm_q0_by_temp[T])
        print(f"所有模式 SOC 曲线已保存（{T}°C）: {out_path}")

    # 功率成分堆叠图（单独一张）
    names = list(all_results.keys())
    names_en = [NAME_EN.get(n, n) for n in names]
    scenario_components = {}
    for n in names:
        mix_dict = all_results[n]["mix"]
        scenario_components[n] = {
            comp: sum(mix_dict.get(act, 0) * activity_components[act][comp] for act in mix_dict)
            for comp in POWER_COMPONENT_ORDER
        }
    with plt.rc_context({"font.family": "Times New Roman", "axes.linewidth": 1.0}):
        fig1, ax1 = plt.subplots(figsize=(8, 5), facecolor="white")
        ax1.set_facecolor("#f5f9fc")
        ax1.tick_params(colors="black", direction="in", which="both")
        for spine in ax1.spines.values():
            spine.set_color("#90a4ae")
        ax1.yaxis.grid(True, linestyle="--", color="#b0bec5", linewidth=0.8)
        ax1.set_axisbelow(True)
        bottom1 = np.zeros(len(names))
        for comp in POWER_COMPONENT_ORDER:
            heights = np.array([scenario_components[n][comp] for n in names])
            ax1.bar(
                names_en, heights, bottom=bottom1, width=0.55,
                label=POWER_COMPONENT_LABELS[comp], color=POWER_COMPONENT_COLORS[comp],
                edgecolor="white", linewidth=1.0
            )
            bottom1 += heights
        y_max = float(np.max(bottom1))
        ax1.set_ylim(0, max(y_max * BAR_TOP_MARGIN, y_max + 0.5))
        ax1.set_ylabel("Avg Power (W)", fontsize=14, color="black")
        ax1.set_title("Power by Scenario (by component)", fontsize=16, color="black")
        ax1.tick_params(axis="x", rotation=15, labelsize=12)
        ax1.tick_params(axis="y", labelsize=12)
        ax1.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#90a4ae")
        plt.tight_layout()
        path_power = os.path.join(SCRIPT_DIR, "power_comparison.png")
        plt.savefig(path_power, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()

    # 续航时长图（单独一张）
    with plt.rc_context({"font.family": "Times New Roman", "axes.linewidth": 1.0}):
        fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor="white")
        ax2.set_facecolor("#f5f9fc")
        ax2.tick_params(colors="black", direction="in", which="both")
        for spine in ax2.spines.values():
            spine.set_color("#90a4ae")
        ax2.yaxis.grid(True, linestyle="--", color="#b0bec5", linewidth=0.8)
        ax2.set_axisbelow(True)
        durations = [all_results[n]["duration_h"] for n in names]
        ax2.bar(names_en, durations, width=0.55, color=DURATION_BAR_COLOR, edgecolor="white", linewidth=1.0)
        d_max = max(durations)
        ax2.set_ylim(0, max(d_max * BAR_TOP_MARGIN, d_max + 2))
        ax2.set_ylabel("Duration (h)", fontsize=14, color="black")
        ax2.set_title("Battery Life by Scenario", fontsize=16, color="black")
        ax2.tick_params(axis="x", rotation=15, labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)
        plt.tight_layout()
        path_duration = os.path.join(SCRIPT_DIR, "duration_comparison.png")
        plt.savefig(path_duration, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()

    print(f"\n功率成分图已保存: power_comparison.png")
    print(f"续航图已保存: duration_comparison.png")

    # 老化情况图：Qm/Q0 按行为模式（完整老化模型：dF/dt=-θ₁*D，Qm=Q0*F*y(T)）- 25°C
    qm_q0_by_pattern = compute_qm_over_q0_per_pattern(
        all_results, activity_powers, model, cycles_per_year=CYCLES_PER_YEAR, T_ambient=25.0
    )
    qm_q0_list = [qm_q0_by_pattern[n] for n in names]
    with plt.rc_context({"font.family": "Times New Roman", "axes.linewidth": 1.0}):
        fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor="white")
        ax3.set_facecolor("#f5f9fc")
        ax3.tick_params(colors="black", direction="in", which="both")
        for spine in ax3.spines.values():
            spine.set_color("#90a4ae")
        ax3.yaxis.grid(True, linestyle="--", color="#b0bec5", linewidth=0.8)
        ax3.set_axisbelow(True)
        ax3.bar(names_en, qm_q0_list, width=0.55, color=DURATION_BAR_COLOR, edgecolor="white", linewidth=1.0)
        q_max = max(qm_q0_list)
        q_min = min(qm_q0_list)
        ax3.set_ylim(max(0, q_min - 0.05), min(1.05, q_max * BAR_TOP_MARGIN))
        ax3.axhline(y=1.0, color="#43a047", linestyle="--", linewidth=1.0, alpha=0.8, label="Qm/Q0 = 1 (no aging)")
        ax3.set_ylabel("Qm / Q0", fontsize=14, color="black")
        ax3.set_title("Capacity Ratio by Usage Pattern (aging model, 25°C)", fontsize=16, color="black")
        ax3.tick_params(axis="x", rotation=15, labelsize=12)
        ax3.tick_params(axis="y", labelsize=12)
        ax3.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#90a4ae")
        plt.tight_layout()
        path_aging = os.path.join(SCRIPT_DIR, "aging_ratio.png")
        plt.savefig(path_aging, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()
    print(f"老化比例图已保存: aging_ratio.png")
    
    # 多温度对比图：-20°C, 25°C, 45°C（qm_q0_by_temp 已在前面计算）
    temperatures = [-20, 25, 45]
    temp_colors = ["#3b82f6", "#10b981", "#ef4444"]  # 蓝色（冷）、绿色（常温）、红色（热）
    
    # 绘制分组柱状图
    with plt.rc_context({"font.family": "Times New Roman", "axes.linewidth": 1.0}):
        fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor="white")
        ax4.set_facecolor("#f5f9fc")
        ax4.tick_params(colors="black", direction="in", which="both")
        for spine in ax4.spines.values():
            spine.set_color("#90a4ae")
        ax4.yaxis.grid(True, linestyle="--", color="#b0bec5", linewidth=0.8)
        ax4.set_axisbelow(True)
        
        x = np.arange(len(names))
        bar_width = 0.25
        for i, T in enumerate(temperatures):
            qm_list = [qm_q0_by_temp[T][n] for n in names]
            offset = (i - 1) * bar_width
            ax4.bar(
                x + offset, qm_list, bar_width,
                label=f"{T}°C", color=temp_colors[i],
                edgecolor="white", linewidth=1.0
            )
        
        ax4.axhline(y=1.0, color="#43a047", linestyle="--", linewidth=1.0, alpha=0.8, label="No aging")
        ax4.set_ylabel("Qm / Q0", fontsize=14, color="black")
        ax4.set_title("Capacity Ratio by Temperature and Usage Pattern", fontsize=16, color="black")
        ax4.set_xticks(x)
        ax4.set_xticklabels(names_en, rotation=15, ha="right")
        ax4.tick_params(axis="both", labelsize=12)
        ax4.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#90a4ae")
        ax4.set_ylim(0, 1.1)
        plt.tight_layout()
        path_temp_comparison = os.path.join(SCRIPT_DIR, "aging_temperature_comparison.png")
        plt.savefig(path_temp_comparison, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close()
    print(f"温度对比图已保存: aging_temperature_comparison.png")
    
    # 每个温度单独的老化图
    for T in [-20, 45]:
        qm_q0_temp = qm_q0_by_temp[T]
        qm_list_temp = [qm_q0_temp[n] for n in names]
        
        with plt.rc_context({"font.family": "Times New Roman", "axes.linewidth": 1.0}):
            fig_t, ax_t = plt.subplots(figsize=(8, 5), facecolor="white")
            ax_t.set_facecolor("#f5f9fc")
            ax_t.tick_params(colors="black", direction="in", which="both")
            for spine in ax_t.spines.values():
                spine.set_color("#90a4ae")
            ax_t.yaxis.grid(True, linestyle="--", color="#b0bec5", linewidth=0.8)
            ax_t.set_axisbelow(True)
            
            bar_color = "#3b82f6" if T == -20 else "#ef4444"
            ax_t.bar(names_en, qm_list_temp, width=0.55, color=bar_color, edgecolor="white", linewidth=1.0)
            q_max_t = max(qm_list_temp)
            q_min_t = min(qm_list_temp)
            ax_t.set_ylim(max(0, q_min_t - 0.05), min(1.1, q_max_t * BAR_TOP_MARGIN))
            ax_t.axhline(y=1.0, color="#43a047", linestyle="--", linewidth=1.0, alpha=0.8, label="Qm/Q0 = 1 (no aging)")
            ax_t.set_ylabel("Qm / Q0", fontsize=14, color="black")
            ax_t.set_title(f"Capacity Ratio by Usage Pattern (aging model, {T}°C)", fontsize=16, color="black")
            ax_t.tick_params(axis="x", rotation=15, labelsize=12)
            ax_t.tick_params(axis="y", labelsize=12)
            ax_t.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="#90a4ae")
            plt.tight_layout()
            path_aging_temp = os.path.join(SCRIPT_DIR, f"aging_ratio_{T}C.png")
            plt.savefig(path_aging_temp, dpi=150, facecolor="white", bbox_inches="tight")
            plt.close()
        print(f"老化比例图（{T}°C）已保存: aging_ratio_{T}C.png")

    # 保存结果（含各模式 Qm/Q0，来自完整老化模型，多温度）
    out = {
        "activity_powers_w": {k: round(v, 4) for k, v in activity_powers.items()},
        "patterns": {
            n: {
                "activity_mix": r["mix"],
                "avg_power_w": round(r["avg_power_w"], 4),
                "estimated_duration_h": round(r["duration_h"], 2),
                "Qm_over_Q0_at_25C": round(qm_q0_by_pattern[n], 4),
                "Qm_over_Q0_at_minus20C": round(qm_q0_by_temp[-20][n], 4),
                "Qm_over_Q0_at_45C": round(qm_q0_by_temp[45][n], 4),
            }
            for n, r in all_results.items()
        },
        "battery_capacity_ah": Q0,
        "aging_model": "Qm=Q0*F*y(T), dF/dt=-θ₁*D, 365 cycles/year",
        "temperature_conditions": {
            "-20C": {n: round(qm_q0_by_temp[-20][n], 4) for n in names},
            "25C": {n: round(qm_q0_by_temp[25][n], 4) for n in names},
            "45C": {n: round(qm_q0_by_temp[45][n], 4) for n in names},
        },
    }
    with open(os.path.join(SCRIPT_DIR, "power_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("结果已保存: power_results.json")


if __name__ == "__main__":
    main()
