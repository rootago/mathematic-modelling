"""
SOC prediction using aging/mobile/new/power_model_aging.json.
Typical scenario: screen on, brightness/CPU 50%, WiFi/GPS on, all values from dataset median.

Model alignment with aging/mobile/new/mobile_new.py:
  - Battery/aging: Q0,R0,R1,C1,E0,k,A,B,v1,v2; vocv(S,Q), soc_indicator(S), degradation_stress(T,I,S),
    aged_capacity(F,T,theta2,Q); ODE dS/dt, dUp/dt, dF/dt = -θ₁×D. Same formulas.
  - Power: P = Intercept_base + C1×screen + C2×screen×bright + C3×cpu_usage_freq + C4×throughput
    + C5×wifi_exp + C6×gps_on + C7×gps_quad; same feature construction and lambda.
  - Data: power_model_aging.json (battery_parameters, coefficients, lambda); battery params
    and coefficients used in simulation come from this JSON (produced by mobile_new.py).
  - Difference: PowerModel here takes (params, intercept, lam); mobile_new uses PowerModel(fit_result)
    with fit_result = {'params', 'intercept', 'lambda'}. Same math inside calc_power_from_row.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(CODE_DIR)
MOBILE_NEW_DIR = os.path.join(CODE_DIR, "aging", "mobile", "new")
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Battery/aging model (same as mobile_new) ----------
Q0, R0, R1, C1 = 8.0, 0.05, 0.03, 1000.0
E0, k, A, B = 3.85, 0.02, 0.25, 3.0
v1, v2 = 0.002, 0.8
V_avg = 3.9

# ---------- Activity parameter mapping (from predict_power.py) ----------
ACTIVITY_TABLE = {
    "Standby": {"bright": [0, 0.1], "cpu": [0, 0.1], "network": [0, 0.15], "gps": 0},
    "Call": {"bright": [0.2, 0.45], "cpu": [0.15, 0.3], "network": [0.25, 0.45], "gps": 0},
    "Video": {"bright": [0.55, 0.85], "cpu": [0.45, 0.75], "network": [0.1, 0.85], "gps": 0},
    "Gaming": {"bright": [0.75, 1], "cpu": [0.75, 1], "network": [0.2, 0.65], "gps": 0},
    "Navigation": {"bright": [0.65, 0.95], "cpu": [0.55, 0.85], "network": [0.65, 0.95], "gps": 1},
}

# Usage patterns (activity, proportion)
USAGE_PATTERNS = {
    "Office Worker": [("Navigation", 0.281), ("Standby", 0.448), ("Video", 0.158), ("Gaming", 0.113)],
    "Commute": [("Video", 0.537), ("Standby", 0.463)],
    "Home": [("Gaming", 0.5), ("Video", 0.465), ("Standby", 0.035)],
    "WeChat Biz": [("Video", 0.42), ("Navigation", 0.114), ("Call", 0.336), ("Standby", 0.042)],
    "Gaming Only": [("Gaming", 1.0)],
}

PATTERN_COLORS = {
    "Office Worker": "#5B9BD5",   # Soft blue
    "Commute": "#ED7D31",         # Soft orange
    "Home": "#FFC000",            # Soft yellow
    "WeChat Biz": "#70AD47",      # Soft green
    "Gaming Only": "#C55A9A",     # Soft pink/purple
}


def vocv(S, Q=None):
    S_safe = np.clip(S, 1e-6, 1.0)
    Q_val = Q if Q is not None else Q0
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q_val)


def soc_indicator(S):
    S_pct = S * 100
    return 1.0 if (S_pct > 95 or S_pct < 10) else 0.0


def degradation_stress(T, I_current, S):
    temp_factor = np.exp(0.035 * (T - 25))
    current_factor = 1 + v1 * abs(I_current) + v2 * soc_indicator(S)
    return temp_factor * current_factor


def aged_capacity(F, T, theta2, Q_nominal=None):
    Q_val = Q_nominal if Q_nominal is not None else Q0
    y_T = np.exp(-theta2 * T)
    return Q_val * F * y_T


def activity_to_row(activity_name, stats=None):
    """
    Convert activity parameters to row format for PowerModel.
    Maps [0,1] normalized values to actual ranges using stats.
    """
    act = ACTIVITY_TABLE[activity_name]
    s = stats or {}
    
    # Brightness: 0-255
    b_min, b_max = act["bright"]
    b_norm = (b_min + b_max) / 2
    bright_level = int(np.clip(
        b_norm * (s.get("bright_max", 255) - s.get("bright_min", 0)) + s.get("bright_min", 0),
        0, 255
    ))
    
    # CPU: 0-100
    c_min, c_max = act["cpu"]
    c_norm = (c_min + c_max) / 2
    cpu_usage = np.clip(
        c_norm * (s.get("cpu_max", 100) - s.get("cpu_min", 0)) + s.get("cpu_min", 0),
        0, 100
    )
    
    # Frequency: higher CPU -> higher freq
    freq_min = s.get("freq_min", 600)
    freq_max = s.get("freq_max", 2400)
    freq_avg = c_norm * (freq_max - freq_min) + freq_min
    
    # Throughput MB/s
    n_min, n_max = act["network"]
    n_norm = (n_min + n_max) / 2
    throughput_mbs = np.clip(
        n_norm * (s.get("throughput_max", 5) - s.get("throughput_min", 0)) + s.get("throughput_min", 0),
        0, 10
    )
    throughput_bytes = throughput_mbs * 1e6
    
    # WiFi
    wifi_status = 1 if n_norm > 0.1 else 0
    wifi_intensity = s.get("wifi_int_min", -65) if wifi_status else -100
    
    # GPS
    gps_status = 3 if act["gps"] == 1 else 0
    
    return {
        "screen_status": 1,
        "bright_level": bright_level,
        "cpu_usage": cpu_usage,
        "frequency_core0": freq_avg, "frequency_core1": freq_avg,
        "frequency_core2": freq_avg, "frequency_core3": freq_avg,
        "frequency_core4": freq_avg, "frequency_core5": freq_avg,
        "frequency_core6": freq_avg, "frequency_core7": freq_avg,
        "wifi_rx": throughput_bytes / 4, "wifi_tx": throughput_bytes / 4,
        "mobile_rx": throughput_bytes / 4, "mobile_tx": throughput_bytes / 4,
        "wifi_status": wifi_status,
        "wifi_intensity": wifi_intensity,
        "gps_status": gps_status,
    }


def compute_pattern_power(pattern_name, power_model, stats=None):
    """Compute weighted average power for a usage pattern."""
    mix = USAGE_PATTERNS[pattern_name]
    mix_dict = {}
    for act, pct in mix:
        mix_dict[act] = mix_dict.get(act, 0) + pct
    total = sum(mix_dict.values())
    if total > 0:
        mix_dict = {k: v / total for k, v in mix_dict.items()}
    
    P_avg = 0.0
    for act, pct in mix_dict.items():
        row = activity_to_row(act, stats)
        P_act = power_model.calc_power_from_row(row)
        P_avg += P_act * pct
    return P_avg


class PowerModel:
    """Power model matching mobile_new, with Intercept_base."""
    def __init__(self, params, intercept, lam):
        self.params = params
        self.intercept = intercept
        self.lam = lam

    def calc_power_from_row(self, row):
        screen_on = 1.0 if row.get("screen_status", 0) == 1 else 0.0
        bright_norm = np.clip(row.get("bright_level", 0), 0, 255) / 255.0
        cpu_usage = np.clip(row.get("cpu_usage", 0), 0, 100) / 100.0
        freq_cols = [c for c in (row.keys() if hasattr(row, "keys") else row.index) if str(c).startswith("frequency_core")]
        freq_norm = sum(row.get(c, 0) for c in freq_cols) / (3000.0 * max(len(freq_cols), 1)) if freq_cols else 0.6
        cpu_usage_freq = cpu_usage * freq_norm
        throughput = sum(row.get(c, 0) for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"]) / 1e6
        throughput = np.clip(throughput, 0, 10)
        wifi_on = 1.0 if row.get("wifi_status", 0) == 1 else 0.0
        wifi_int = np.clip(row.get("wifi_intensity", -50), -100, -20)
        wifi_exp = wifi_on * np.exp(-self.lam * wifi_int)
        gps_status = row.get("gps_status", 0)
        gps_on = 1.0 if gps_status > 0 else 0.0
        gps_act_norm = np.clip(gps_status, 0, 3) / 3.0 if gps_status > 0 else 0.0
        gps_quad = gps_on * (gps_act_norm ** 2)
        features = [screen_on, screen_on * bright_norm, cpu_usage_freq, throughput, wifi_exp, gps_on, gps_quad]
        P = self.intercept + sum(p * f for p, f in zip(self.params, features))
        return max(P, 0.1)


def load_mobile_data(max_files=20):
    """Load mobile data for median computation."""
    all_dfs = []
    folders = sorted(
        [d for d in os.listdir(DATA_DIR)
         if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()]
    )[:max_files]
    for folder in folders:
        pattern = os.path.join(DATA_DIR, folder, "*", "*_dynamic_processed.csv")
        for f in glob.glob(pattern):
            try:
                df = pd.read_csv(f, low_memory=False)
                all_dfs.append(df)
            except Exception:
                pass
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def compute_median_scenario(data):
    """
    Compute typical scenario: all conditions ON, values from dataset median.
    """
    df = data[data["battery_charging_status"] == 3].copy()  # discharge only
    if len(df) < 100:
        df = data.copy()

    freq_cols = [c for c in df.columns if c.startswith("frequency_core")]
    tp_cols = [c for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"] if c in df.columns]

    # Brightness median (when screen on)
    screen_on = df["screen_status"] == 1
    bright_median = df.loc[screen_on, "bright_level"].dropna().clip(0, 255).median()
    if np.isnan(bright_median):
        bright_median = 127.5

    # CPU usage median
    cpu_median = df["cpu_usage"].dropna().clip(0, 100).median()
    if np.isnan(cpu_median):
        cpu_median = 50.0

    # Freq norm median: sum(freq)/(3000*n_cores)
    if freq_cols:
        freq_sum = df[freq_cols].sum(axis=1)
        n_cores = len(freq_cols)
        freq_norm_vals = freq_sum / (3000.0 * n_cores)
        freq_norm_median = freq_norm_vals.median()
    else:
        freq_norm_median = 0.6
    if np.isnan(freq_norm_median):
        freq_norm_median = 0.6

    # Throughput median (MB/s)
    if tp_cols:
        tp = df[tp_cols].sum(axis=1).clip(0, 1e7) / 1e6
        throughput_median = tp[tp > 0].median()
    else:
        throughput_median = 0.0
    if np.isnan(throughput_median):
        throughput_median = 0.1

    # WiFi intensity median (when WiFi on)
    wifi_on = df["wifi_status"] == 1
    if "wifi_intensity" in df.columns and wifi_on.any():
        wifi_int = df.loc[wifi_on, "wifi_intensity"].dropna().clip(-100, -20)
        wifi_int_median = wifi_int.median()
    else:
        wifi_int_median = -50
    if np.isnan(wifi_int_median):
        wifi_int_median = -50

    # GPS status median when GPS on (0-3)
    gps_on = df["gps_status"] > 0
    if gps_on.any():
        gps_median = df.loc[gps_on, "gps_status"].clip(0, 3).median()
    else:
        gps_median = 2
    if np.isnan(gps_median):
        gps_median = 2

    # Temperature median
    if "battery_temperature" in df.columns:
        T_median = df["battery_temperature"].dropna().median()
    else:
        T_median = 28.0
    if np.isnan(T_median):
        T_median = 28.0

    scenario = {
        "screen_status": 1,
        "bright_level": float(bright_median),
        "cpu_usage": float(cpu_median),
        "freq_norm": float(freq_norm_median),
        "throughput_MB_s": float(throughput_median),
        "wifi_status": 1,
        "wifi_intensity": float(wifi_int_median),
        "gps_status": int(round(gps_median)),
        "battery_temperature": float(T_median),
    }
    return scenario


def scenario_to_row(scenario):
    """Convert scenario dict to row format for PowerModel."""
    row = {
        "screen_status": scenario["screen_status"],
        "bright_level": scenario["bright_level"],
        "cpu_usage": scenario["cpu_usage"],
        "wifi_status": scenario["wifi_status"],
        "wifi_intensity": scenario["wifi_intensity"],
        "gps_status": scenario["gps_status"],
        "battery_temperature": scenario["battery_temperature"],
    }
    freq_cols = [f"frequency_core{i}" for i in range(8)]
    n_cores = 8
    # freq_norm = sum / (3000 * n_cores) => sum = freq_norm * 3000 * n_cores
    total_freq = scenario["freq_norm"] * 3000 * n_cores
    per_core = total_freq / n_cores
    for i, c in enumerate(freq_cols):
        row[c] = per_core

    tp = scenario["throughput_MB_s"] * 1e6
    row["wifi_rx"] = tp / 4
    row["wifi_tx"] = tp / 4
    row["mobile_rx"] = tp / 4
    row["mobile_tx"] = tp / 4
    return row


def simulate_soc_constant_power(power_model, scenario_row, Q0, R0, R1, C1_cap,
                                theta1=1e-8, theta2=0.001, T_const=28.0,
                                t_max_hours=24 * 2, Q0_override=None):
    """
    Simulate SOC curve at constant power, 100% to 0%.
    Q0_override: if provided, overrides Q0 for capacity sensitivity.
    """
    Q_use = Q0_override if Q0_override is not None else Q0
    P_const = power_model.calc_power_from_row(scenario_row)
    t_span_sec = t_max_hours * 3600
    S0, Up0, F0 = 1.0, 0.0, 1.0

    def event_soc_zero(t, y):
        return y[0] - 0.01
    event_soc_zero.terminal = True
    event_soc_zero.direction = -1

    def odefun(t, y):
        S_val = np.clip(y[0], 1e-6, 1.0)
        Up_val = y[1]
        F_val = np.clip(y[2], 0.5, 1.0)
        if S_val <= 1e-6:
            return np.array([0.0, 0.0, 0.0])
        P = P_const
        T = T_const
        Qm = aged_capacity(F_val, T, theta2, Q_use)
        Qm = max(Qm, Q_use * 0.5)
        Vocv_val = vocv(S_val, Qm)
        V_eff = Vocv_val - Up_val
        disc = V_eff**2 - 4 * R0 * P
        I_val = (V_eff - np.sqrt(max(disc, 0))) / (2 * R0) if disc >= 0 else V_eff / (2 * R0)
        I_val = max(I_val, 0.0)
        D = degradation_stress(T, I_val, S_val)
        dS_dt = -I_val / (Qm * 3600.0)
        dUp_dt = -Up_val / (R1 * C1_cap) + I_val / C1_cap
        dF_dt = -theta1 * D
        return np.array([dS_dt, dUp_dt, dF_dt])

    sol = solve_ivp(
        odefun, (0, t_span_sec), [S0, Up0, F0],
        method="RK45", dense_output=True, max_step=60,
        events=event_soc_zero
    )
    t_sec = sol.t
    soc_pct = np.clip(sol.y[0], 0, 1) * 100
    return t_sec / 60, soc_pct, P_const


# ---------- Sensitivity analysis: +/-5%, +/-10%, +/-20% ----------
PERTURB_PCTS = [-20, -10, -5, 0, 5, 10, 20]  # 0 = baseline

SENSITIVITY_PARAMS = [
    ("Q0", "Battery Capacity Q0 (Ah)", "battery", None, 8.0),
    ("C1_screen", "Screen Coef C1", "power", 0, None),
    ("C2_bright", "Brightness Coef C2", "power", 1, None),
    ("C3_cpu_usage_freq", "CPU Coef C3", "power", 2, None),
    ("C4_throughput", "Network Coef C4", "power", 3, None),
    ("C5_wifi", "WiFi Coef C5", "power", 4, None),
    ("T", "Temperature T (C)", "temperature", None, None),
    ("C6_gps", "GPS Coef C6", "power", 5, None),
    ("C7_gps_quad", "GPS Quad Coef C7", "power", 6, None),
    ("theta1", "θ₁ decay factor (dF/dt = -θ₁×D)", "theta1", None, None),
    ("theta2", "θ₂ temperature decay (y(T)=exp(-θ₂×T))", "theta2", None, None),
]


def run_sensitivity_analysis(power_model, scenario_row, scenario, bp, coef, lam, stats,
                             theta1=1e-8, theta2=0.001, t_max_hours=48):
    """
    Perturb each parameter by +/-5%, +/-10%, +/-20%; plot time-to-empty vs param change.
    X-axis: parameter change amount (%). Y-axis: time to empty (h).
    Only show WeChat Biz and Commute patterns.
    Q0 and Temperature: separate plots
    Other power coefficients: combined into one 3x3 subplot figure
    """
    base_params = np.array([
        coef["C1_screen"], coef["C2_bright"], coef["C3_cpu_usage_freq"],
        coef["C4_throughput"], coef["C5_wifi"], coef["C6_gps"], coef["C7_gps_quad"]
    ])
    base_Q0 = bp["Q0_Ah"]
    base_T = scenario.get("battery_temperature", 28.0)

    sens_dir = os.path.join(OUTPUT_DIR, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    # Only keep WeChat Biz and Commute
    selected_patterns = ["WeChat Biz", "Commute"]
    
    # Separate params into standalone and combined
    standalone_params = []  # Q0, T, θ₁, θ₂
    combined_params = []     # All power coefficients
    
    for param_info in SENSITIVITY_PARAMS:
        param_id, param_label, param_type, idx, base_val = param_info
        if param_type in ["battery", "temperature", "theta1", "theta2"]:
            standalone_params.append(param_info)
        else:
            combined_params.append(param_info)
    
    # Store results for all parameters
    all_results = {}
    
    for param_id, param_label, param_type, idx, base_val in SENSITIVITY_PARAMS:
        if param_type == "battery":
            base_val = base_Q0
        elif param_type == "temperature":
            base_val = base_T
        elif param_type == "power":
            base_val = base_params[idx]
        elif param_type == "theta1":
            base_val = theta1
        elif param_type == "theta2":
            base_val = theta2

        # Compute time-to-empty for selected patterns at each perturbation
        pattern_results = {name: [] for name in selected_patterns}
        
        for pct in PERTURB_PCTS:
            mult = 1.0 + pct / 100.0
            
            # Create modified power model / parameters
            if param_type == "battery":
                Q_use = base_Q0 * mult
                pm = power_model
            elif param_type == "temperature":
                Q_use = base_Q0
                pm = power_model
            elif param_type == "theta1":
                # θ₁: run ODE simulation (dF/dt = -θ₁×D), same t_empty for both patterns (typical scenario)
                theta1_pert = theta1 * mult
                t_min, soc_pct, P_const = simulate_soc_constant_power(
                    power_model, scenario_row,
                    Q0=base_Q0, R0=bp["R0_ohm"], R1=bp["R1_ohm"], C1_cap=bp["C1_F"],
                    theta1=theta1_pert, theta2=theta2, T_const=base_T, t_max_hours=t_max_hours
                )
                t_empty_h = t_min[-1] / 60.0
                for pattern_name in selected_patterns:
                    pattern_results[pattern_name].append(t_empty_h)
                continue
            elif param_type == "theta2":
                # θ₂: analytic energy = Q0*V*exp(-θ₂*T), different P per pattern
                Q_use = base_Q0
                pm = power_model
            else:  # power coefficients
                new_params = base_params.copy()
                new_params[idx] = base_params[idx] * mult
                pm = PowerModel(new_params, coef["Intercept_base"], lam)
                Q_use = base_Q0
            
            if param_type != "theta1":
                # For each selected pattern, compute time to empty
                for pattern_name in selected_patterns:
                    P_avg = compute_pattern_power(pattern_name, pm, stats)
                    # Time to empty = E / P = Q * V / P
                    energy_wh = Q_use * V_avg
                    if param_type == "temperature":
                        T_factor = np.exp(-theta2 * base_T * mult)
                        energy_wh *= T_factor
                    elif param_type == "theta2":
                        theta2_pert = theta2 * mult
                        energy_wh *= np.exp(-theta2_pert * base_T)
                    t_empty_h = energy_wh / max(P_avg, 0.1)
                    pattern_results[pattern_name].append(t_empty_h)
        
        all_results[param_id] = {
            "label": param_label,
            "type": param_type,
            "base_val": base_val,
            "results": pattern_results
        }

    # Plot standalone parameters (Q0 and Temperature) individually
    for param_id, param_label, param_type, idx, base_val in standalone_params:
        fig, ax = plt.subplots(figsize=(10, 7))
        x_vals = np.array(PERTURB_PCTS)
        
        res = all_results[param_id]
        for pattern_name in selected_patterns:
            y_vals = np.array(res["results"][pattern_name])
            color = PATTERN_COLORS[pattern_name]
            ax.plot(x_vals, y_vals, marker="o", markersize=7, linewidth=2, 
                   color=color, label=pattern_name, alpha=0.85)
        
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Parameter Change Amount (%)", fontsize=18)
        ax.set_ylabel("Time to Empty (h)", fontsize=18)
        val_str = f"{res['base_val']:.2f}" if isinstance(res['base_val'], float) else f"{res['base_val']:.1f}"
        ax.set_title(f"Sensitivity: {param_label}\n(Baseline {param_id}={val_str})", fontsize=20, pad=12)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{pct}%" for pct in x_vals], fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(loc="best", fontsize=14)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(x_vals.min() - 2, x_vals.max() + 2)
        
        # Compute y-axis limits from selected patterns only
        y_all = np.concatenate([res["results"][p] for p in selected_patterns])
        y_max = y_all.max()
        y_min_data = y_all.min()
        y_range = y_max - y_min_data
        y_pad = max(y_range * 0.15, y_max * 0.06, 0.8)
        ax.set_ylim(max(0, y_min_data - y_pad), y_max + y_pad)
        fig.tight_layout(pad=1.8)
        out_name = f"sensitivity_{param_id}.png"
        fig.savefig(os.path.join(sens_dir, out_name), dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        
        t_range_str = f"{y_all.min():.1f}h ~ {y_all.max():.1f}h"
        print(f"  Saved: {out_name} (T_empty range: {t_range_str})")

    # Plot combined figure: T, θ₁, θ₂ on one chart (same style as sensitivity_power_coefficients_combined)
    # Both scenarios: WeChat Biz (solid + circle), Commute (dashed + square)
    temp_aging_params = [p for p in SENSITIVITY_PARAMS if p[0] in ("T", "theta1", "theta2")]
    x_vals = np.array(PERTURB_PCTS)
    n_ta = len(temp_aging_params)
    ta_colors = plt.cm.tab10(np.linspace(0, 1, max(n_ta, 10)))[:n_ta]
    ta_short = {"T": "T", "theta1": "θ₁", "theta2": "θ₂"}
    with plt.rc_context({"font.family": "Times New Roman"}):
        fig_ta, ax_ta = plt.subplots(figsize=(14, 9))
        for i, (param_id, param_label, param_type, idx, base_val) in enumerate(temp_aging_params):
            res = all_results[param_id]
            short = ta_short.get(param_id, param_id)
            for pattern_name in selected_patterns:
                y_vals = np.array(res["results"][pattern_name])
                linestyle = "-" if pattern_name == "WeChat Biz" else "--"
                marker = "o" if pattern_name == "WeChat Biz" else "s"
                label = f"{short} ({pattern_name})"
                ax_ta.plot(x_vals, y_vals, marker=marker, markersize=4, linewidth=1.8,
                           color=ta_colors[i], linestyle=linestyle,
                           label=label, alpha=0.85)
        ax_ta.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax_ta.set_xlabel("Parameter Change Amount (%)", fontsize=20)
        ax_ta.set_ylabel("Time to Empty (h)", fontsize=20)
        ax_ta.set_title("Sensitivity: T, θ₁, θ₂\n(WeChat Biz & Commute)", fontsize=22, fontweight="bold", pad=12)
        ax_ta.set_xticks(x_vals)
        ax_ta.set_xticklabels([f"{pct}%" for pct in x_vals], fontsize=18)
        ax_ta.tick_params(axis="both", which="major", labelsize=18)
        ax_ta.legend(loc="best", fontsize=15, ncol=2)
        ax_ta.grid(True, alpha=0.4)
        ax_ta.set_xlim(x_vals.min() - 2, x_vals.max() + 2)
        y_ta_list = []
        for pid, _, _, _, _ in temp_aging_params:
            for pn in selected_patterns:
                y_ta_list.extend(all_results[pid]["results"][pn])
        y_ta = np.array(y_ta_list)
        y_max_ta = y_ta.max()
        y_min_ta = y_ta.min()
        y_pad_ta = max((y_max_ta - y_min_ta) * 0.15, y_max_ta * 0.06, 0.5)
        ax_ta.set_ylim(max(0, y_min_ta - y_pad_ta), y_max_ta + y_pad_ta)
        fig_ta.tight_layout(pad=1.8)
        fig_ta.savefig(os.path.join(sens_dir, "sensitivity_T_theta1_theta2_combined.png"), dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig_ta)
        print(f"  Saved: sensitivity_T_theta1_theta2_combined.png (T, θ₁, θ₂, WeChat Biz & Commute)")

    # ---------- One figure: Temperature, C1, C3 三图从左到右排列，五个场景的线都画上 ----------
    all_five_patterns = list(USAGE_PATTERNS.keys())
    trio_params_info = [
        ("T", "Temperature T (°C)", "temperature", None),
        ("C1_screen", "Screen Coef C1", "power", 0),
        ("C3_cpu_usage_freq", "CPU Coef C3", "power", 2),
    ]
    x_vals = np.array(PERTURB_PCTS)
    trio_results = {}
    for param_id, param_label, param_type, power_idx in trio_params_info:
        pattern_results_five = {name: [] for name in all_five_patterns}
        for pct in PERTURB_PCTS:
            mult = 1.0 + pct / 100.0
            if param_type == "temperature":
                Q_use = base_Q0
                pm = power_model
                T_factor = np.exp(-theta2 * base_T * mult)
                for pattern_name in all_five_patterns:
                    P_avg = compute_pattern_power(pattern_name, pm, stats)
                    energy_wh = base_Q0 * V_avg * T_factor
                    t_empty_h = energy_wh / max(P_avg, 0.1)
                    pattern_results_five[pattern_name].append(t_empty_h)
            else:
                new_params = base_params.copy()
                new_params[power_idx] = base_params[power_idx] * mult
                pm = PowerModel(new_params, coef["Intercept_base"], lam)
                Q_use = base_Q0
                for pattern_name in all_five_patterns:
                    P_avg = compute_pattern_power(pattern_name, pm, stats)
                    energy_wh = Q_use * V_avg
                    t_empty_h = energy_wh / max(P_avg, 0.1)
                    pattern_results_five[pattern_name].append(t_empty_h)
        trio_results[param_id] = {"label": param_label, "base_val": all_results[param_id]["base_val"], "results": pattern_results_five}
    with plt.rc_context({"font.family": "Times New Roman"}):
        fig_trio, axes_trio = plt.subplots(1, 3, figsize=(18, 6))
        for col, (param_id, param_label, param_type, _) in enumerate(trio_params_info):
            ax = axes_trio[col]
            res = trio_results[param_id]
            for pattern_name in all_five_patterns:
                y_vals = np.array(res["results"][pattern_name])
                color = PATTERN_COLORS[pattern_name]
                ax.plot(x_vals, y_vals, marker="o", markersize=5, linewidth=1.8,
                        color=color, label=pattern_name, alpha=0.85)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Parameter Change Amount (%)", fontsize=20)
            ax.set_ylabel("Time to Empty (h)", fontsize=20)
            val_str = f"{res['base_val']:.2f}" if isinstance(res['base_val'], float) else f"{res['base_val']:.1f}"
            ax.set_title(f"{param_label}\n(Baseline {param_id}={val_str})", fontsize=18, pad=8)
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f"{pct}%" for pct in x_vals], fontsize=16)
            ax.tick_params(axis="y", labelsize=16)
            ax.legend(loc="best", fontsize=14)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(x_vals.min() - 2, x_vals.max() + 2)
            y_all = np.concatenate([res["results"][p] for p in all_five_patterns])
            y_max = y_all.max()
            y_min_data = y_all.min()
            y_range = y_max - y_min_data
            y_pad = max(y_range * 0.15, y_max * 0.06, 0.5)
            ax.set_ylim(max(0, y_min_data - y_pad), y_max + y_pad)
        fig_trio.suptitle("Sensitivity: Temperature, C1, C3 (All 5 Scenarios)", fontsize=22, fontweight="bold", y=1.02)
        fig_trio.tight_layout(pad=1.2)
        fig_trio.savefig(os.path.join(sens_dir, "sensitivity_T_C1_C3_combined.png"), dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig_trio)
        print(f"  Saved: sensitivity_T_C1_C3_combined.png (Temperature, C1, C3, all 5 scenarios)")

    # Plot combined figure: all power coefficients on one chart, both usage patterns
    x_vals = np.array(PERTURB_PCTS)
    
    # Define colors for different parameters
    param_colors = plt.cm.tab10(np.linspace(0, 1, len(combined_params)))
    
    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(18, 12))
        
        for i, (param_id, param_label, param_type, idx, base_val) in enumerate(combined_params):
            res = all_results[param_id]
            
            # Shortened labels
            short_label = param_label.replace(" Coef", "").replace("Brightness", "Bright")
            
            # Plot both patterns for this parameter
            for pattern_name in selected_patterns:
                y_vals = np.array(res["results"][pattern_name])
                
                # Use solid line for WeChat Biz, dashed line for Commute
                linestyle = '-' if pattern_name == "WeChat Biz" else '--'
                marker = 'o' if pattern_name == "WeChat Biz" else 's'
                
                label = f"{short_label} ({pattern_name})"
                
                ax.plot(x_vals, y_vals, marker=marker, markersize=4, linewidth=1.8, 
                       color=param_colors[i], linestyle=linestyle,
                       label=label, alpha=0.85)
        
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Parameter Change Amount (%)", fontsize=24)
        ax.set_ylabel("Time to Empty (h)", fontsize=24)
        ax.set_title("Power Coefficient Sensitivity Analysis\n(WeChat Biz & Commute)", 
                     fontsize=26, fontweight='bold', pad=12)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{pct}%" for pct in x_vals], fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.legend(loc="best", fontsize=18, ncol=2)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(x_vals.min() - 2, x_vals.max() + 2)
        
        # Compute y-axis limits from all results
        y_all_list = []
        for param_id, _, _, _, _ in combined_params:
            for pattern_name in selected_patterns:
                y_all_list.extend(all_results[param_id]["results"][pattern_name])
        y_all = np.array(y_all_list)
        y_max = y_all.max()
        y_min_data = y_all.min()
        y_range = y_max - y_min_data
        y_pad = max(y_range * 0.15, y_max * 0.06, 0.5)
        ax.set_ylim(max(0, y_min_data - y_pad), y_max + y_pad)
        
        fig.tight_layout(pad=1.8)
        out_combined = "sensitivity_power_coefficients_combined.png"
        fig.savefig(os.path.join(sens_dir, out_combined), dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(f"  Saved: {out_combined} (all power coefficients, both patterns)")

    # ---------- One figure: ALL parameters (Q0, T, C1-C7), same style as combined ----------
    _plot_sensitivity_all_params(all_results, SENSITIVITY_PARAMS, selected_patterns, sens_dir)

    # ---------- Radar chart: C1,C2,C3,C5 only (exclude Q0, T, C4, C6, C7), same two scenarios ----------
    radar_params = [p for p in combined_params if p[0] not in ("C4_throughput", "C6_gps", "C7_gps_quad")]
    _plot_radar_chart(all_results, radar_params, selected_patterns, sens_dir)


def _plot_sensitivity_all_params(all_results, param_list, selected_patterns, sens_dir):
    """
    One sensitivity figure: ALL parameters (Q0, T, C1-C7). X = param change (%), Y = time to empty (h).
    Same style as power_coefficients_combined: one color per param, solid/dashed for WeChat Biz/Commute.
    """
    x_vals = np.array(PERTURB_PCTS)
    n_params = len(param_list)
    param_colors = plt.cm.tab10(np.linspace(0, 1, max(n_params, 10)))[:n_params]

    short_labels = {
        "Q0": "Q0",
        "T": "T",
        "C1_screen": "C1",
        "C2_bright": "C2",
        "C3_cpu_usage_freq": "C3",
        "C4_throughput": "C4",
        "C5_wifi": "C5",
        "C6_gps": "C6",
        "C7_gps_quad": "C7",
        "theta1": "θ₁",
        "theta2": "θ₂",
    }

    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(16, 10))

        for i, (param_id, param_label, param_type, idx, base_val) in enumerate(param_list):
            res = all_results[param_id]
            short = short_labels.get(param_id, param_id)

            for pattern_name in selected_patterns:
                y_vals = np.array(res["results"][pattern_name])
                linestyle = "-" if pattern_name == "WeChat Biz" else "--"
                marker = "o" if pattern_name == "WeChat Biz" else "s"
                label = f"{short} ({pattern_name})"
                ax.plot(
                    x_vals, y_vals, marker=marker, markersize=4, linewidth=1.8,
                    color=param_colors[i], linestyle=linestyle, label=label, alpha=0.85
                )

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Parameter Change Amount (%)", fontsize=20)
        ax.set_ylabel("Time to Empty (h)", fontsize=20)
        ax.set_title("Sensitivity Analysis: All Parameters (Q0, T, C1–C7, θ₁, θ₂)\n(WeChat Biz & Commute)", fontsize=22, fontweight="bold", pad=12)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{pct}%" for pct in x_vals], fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13, frameon=True)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(x_vals.min() - 2, x_vals.max() + 2)

        y_all_list = []
        for param_id, _, _, _, _ in param_list:
            for pattern_name in selected_patterns:
                y_all_list.extend(all_results[param_id]["results"][pattern_name])
        y_all = np.array(y_all_list)
        y_max = y_all.max()
        y_min_data = y_all.min()
        y_range = y_max - y_min_data
        y_pad = max(y_range * 0.12, y_max * 0.06, 0.5)
        ax.set_ylim(max(0, y_min_data - y_pad), y_max + y_pad)

        fig.tight_layout(pad=1.8, rect=(0, 0.14, 1, 1))
        out_path = os.path.join(sens_dir, "sensitivity_all_params.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(f"  Saved: sensitivity_all_params.png (all params Q0,T,C1-C7, both patterns)")


def _plot_radar_chart(all_results, param_list, selected_patterns, sens_dir):
    """
    Plot radar (spider) chart: axes = selected power params (e.g. C1,C2,C3,C5), 2 series = WeChat Biz & Commute.
    Value on each axis = sensitivity range (max - min T_empty when param varies ±20%),
    normalized to 0-14 scale. Filled polygon per scenario like reference.
    """
    # Short axis labels (C1, C2, C3, C5 when C4,C6,C7 excluded)
    short_labels = {
        "C1_screen": "C1",
        "C2_bright": "C2",
        "C3_cpu_usage_freq": "C3",
        "C4_throughput": "C4",
        "C5_wifi": "C5",
        "C6_gps": "C6",
        "C7_gps_quad": "C7",
    }

    n_params = len(param_list)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])  # close polygon

    # Per (param, pattern): sensitivity range = max(T_empty) - min(T_empty)
    all_ranges = []
    pattern_values = {name: [] for name in selected_patterns}
    labels = []

    for param_id, _, _, _, _ in param_list:
        res = all_results[param_id]
        labels.append(short_labels.get(param_id, param_id))
        for pattern_name in selected_patterns:
            ts = np.array(res["results"][pattern_name])
            r = float(np.max(ts) - np.min(ts))
            pattern_values[pattern_name].append(r)
            all_ranges.append(r)

    v_max = max(all_ranges) if all_ranges else 1.0
    # Normalize to 0–14 scale (like reference)
    scale_max = 14.0
    values_norm = {name: np.array(pattern_values[name]) / v_max * scale_max for name in selected_patterns}

    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(projection="polar"))

        for pattern_name in selected_patterns:
            vals = values_norm[pattern_name]
            vals_closed = np.concatenate([vals, [vals[0]]])
            color = PATTERN_COLORS[pattern_name]
            ax.fill(angles_closed, vals_closed, color=color, alpha=0.35)
            ax.plot(angles_closed, vals_closed, color=color, linewidth=2, marker="o", markersize=6, label=pattern_name)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=16)
        ax.set_ylim(0, scale_max)
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_yticklabels(["0", "2", "4", "6", "8", "10", "12", "14"], fontsize=14)
        ax.set_title("Power Coefficients Sensitivity (WeChat Biz & Commute)\nRadar: sensitivity range (±20%)", fontsize=18, fontweight="bold", pad=24)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.0), fontsize=15)
        ax.grid(True, alpha=0.5)
        fig.tight_layout(pad=2.0)
        out_radar = os.path.join(sens_dir, "sensitivity_radar_power_params.png")
        fig.savefig(out_radar, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  Saved: sensitivity_radar_power_params.png (radar, power coeffs, both scenarios)")


def main():
    print("=" * 60)
    print("Typical Scenario SOC Prediction (All ON, Median Values)")
    print("=" * 60)

    json_path = os.path.join(MOBILE_NEW_DIR, "power_model_aging.json")
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bp = cfg["battery_parameters"]
    coef = cfg["coefficients"]
    lam = cfg["lambda"]

    params = np.array([
        coef["C1_screen"], coef["C2_bright"], coef["C3_cpu_usage_freq"],
        coef["C4_throughput"], coef["C5_wifi"], coef["C6_gps"], coef["C7_gps_quad"]
    ])
    power_model = PowerModel(params, coef["Intercept_base"], lam)

    print("\nLoading mobile data for stats and median...")
    data = load_mobile_data(max_files=25)
    if data is None:
        print("Data load failed, using defaults")
        scenario = {"battery_temperature": 28.0}
        stats = {
            "bright_min": 0, "bright_max": 255,
            "cpu_min": 0, "cpu_max": 100,
            "freq_min": 600, "freq_max": 2400,
            "throughput_min": 0, "throughput_max": 5,
            "wifi_int_min": -65,
        }
    else:
        scenario = compute_median_scenario(data)
        print("Typical scenario (median values):")
        for k, v in scenario.items():
            print(f"  {k}: {v}")
        
        # Compute stats for usage patterns
        stats = {}
        df = data[data["battery_charging_status"] == 3].copy() if "battery_charging_status" in data.columns else data
        if len(df) < 100:
            df = data.copy()
        
        if "bright_level" in df.columns:
            br = pd.to_numeric(df["bright_level"], errors="coerce").dropna()
            stats["bright_min"] = br.quantile(0.05)
            stats["bright_max"] = br.quantile(0.95)
        if "cpu_usage" in df.columns:
            cu = pd.to_numeric(df["cpu_usage"], errors="coerce").dropna()
            stats["cpu_min"] = cu.quantile(0.05)
            stats["cpu_max"] = cu.quantile(0.95)
        freq_cols = [c for c in df.columns if c.startswith("frequency_core")]
        if freq_cols:
            freq = df[freq_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            stats["freq_min"] = freq.quantile(0.05)
            stats["freq_max"] = freq.quantile(0.95)
        tp_cols = [c for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"] if c in df.columns]
        if tp_cols:
            tp = df[tp_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) / 1e6
            tp = tp.clip(0, 10)
            stats["throughput_min"] = tp.quantile(0.05)
            stats["throughput_max"] = tp.quantile(0.95)
        if "wifi_intensity" in df.columns:
            wi = pd.to_numeric(df["wifi_intensity"], errors="coerce").dropna()
            wi = wi[(wi >= -100) & (wi <= -20)]
            if len(wi) > 0:
                stats["wifi_int_min"] = wi.quantile(0.05)

    scenario_row = scenario_to_row(scenario)
    T_const = scenario["battery_temperature"]

    theta1, theta2 = 1e-8, 0.001
    t_min, soc_pct, P_avg = simulate_soc_constant_power(
        power_model, scenario_row,
        Q0=bp["Q0_Ah"], R0=bp["R0_ohm"], R1=bp["R1_ohm"], C1_cap=bp["C1_F"],
        theta1=theta1, theta2=theta2, T_const=T_const, t_max_hours=48
    )

    t_empty_h = t_min[-1] / 60
    print(f"\nConstant power: P = {P_avg:.3f} W")
    print(f"100% -> 0% predicted duration: {t_empty_h:.2f} h ({t_empty_h/24:.2f} days)")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(t_min, soc_pct, "b-", linewidth=2, label="SOC (Typical Scenario)")
    ax.set_xlabel("Time (min)", fontsize=18)
    ax.set_ylabel("SOC (%)", fontsize=18)
    ax.set_title(
        f"SOC Prediction: Typical Scenario (All ON, Median Values)\n"
        f"P={P_avg:.2f}W, Time to empty ~{t_empty_h:.1f}h",
        fontsize=20, pad=12
    )
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    fig.tight_layout(pad=1.8)
    out_path = os.path.join(OUTPUT_DIR, "soc_typical_scenario.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    # Save scenario and SOC data
    result_path = os.path.join(OUTPUT_DIR, "typical_scenario_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenario": scenario,
            "P_avg_W": float(P_avg),
            "time_to_empty_h": float(t_empty_h),
            "t_min": t_min.tolist(),
            "soc_pct": soc_pct.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {result_path}")

    # ---------- Output actual parameters for WeChat Biz and Commute ----------
    print("\n" + "=" * 60)
    print("Computing actual parameters for WeChat Biz & Commute")
    print("=" * 60)
    
    selected_patterns = ["WeChat Biz", "Commute"]
    pattern_params = []
    
    for pattern_name in selected_patterns:
        mix = USAGE_PATTERNS[pattern_name]
        # Normalize proportions
        total = sum(pct for _, pct in mix)
        mix_normalized = [(act, pct / total) for act, pct in mix]
        
        # Compute weighted average parameters
        weighted_params = {
            "screen_status": 0.0,
            "bright_level": 0.0,
            "cpu_usage": 0.0,
            "frequency_avg": 0.0,
            "throughput_MB_s": 0.0,
            "wifi_status": 0.0,
            "wifi_intensity": 0.0,
            "gps_status": 0.0,
        }
        
        for act_name, pct in mix_normalized:
            row = activity_to_row(act_name, stats)
            
            # Calculate average values
            freq_avg = (row["frequency_core0"] + row["frequency_core1"] + 
                       row["frequency_core2"] + row["frequency_core3"]) / 4
            throughput_mb = (row["wifi_rx"] + row["wifi_tx"] + 
                            row["mobile_rx"] + row["mobile_tx"]) / 1e6
            
            # Accumulate weighted averages
            weighted_params["screen_status"] += row["screen_status"] * pct
            weighted_params["bright_level"] += row["bright_level"] * pct
            weighted_params["cpu_usage"] += row["cpu_usage"] * pct
            weighted_params["frequency_avg"] += freq_avg * pct
            weighted_params["throughput_MB_s"] += throughput_mb * pct
            weighted_params["wifi_status"] += row["wifi_status"] * pct
            weighted_params["wifi_intensity"] += row["wifi_intensity"] * pct
            weighted_params["gps_status"] += row["gps_status"] * pct
        
        # Store weighted average for this pattern (only selected parameters)
        pattern_params.append({
            "Usage Pattern": pattern_name,
            "Brightness\n(0-255)": f"{weighted_params['bright_level']:.1f}",
            "CPU Usage\n(%)": f"{weighted_params['cpu_usage']:.1f}",
            "Frequency\n(MHz)": f"{weighted_params['frequency_avg']:.0f}",
            "Throughput\n(MB/s)": f"{weighted_params['throughput_MB_s']:.3f}",
            "WiFi Status\n(0/1)": f"{weighted_params['wifi_status']:.2f}",
            "WiFi Intensity\n(dBm)": f"{weighted_params['wifi_intensity']:.1f}",
            "GPS Status\n(0/3)": f"{weighted_params['gps_status']:.2f}",
        })
    
    # Create DataFrame
    df_params = pd.DataFrame(pattern_params)
    
    # Save to CSV
    params_csv_path = os.path.join(OUTPUT_DIR, "usage_pattern_actual_parameters.csv")
    df_params.to_csv(params_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nActual parameters saved to: {params_csv_path}")
    
    # Create table image with better styling
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [df_params.columns.tolist()] + df_params.values.tolist()
    
    # Create table with adjusted column widths
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.14, 0.13, 0.12, 0.13, 0.13, 0.12, 0.13, 0.10])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3.0)
    
    # Modern color palette
    header_color = '#2E5090'      # Deep blue
    row1_color = '#E8F0F8'        # Light blue
    row2_color = '#F5F8FB'        # Very light blue
    
    # Header styling - Times New Roman, no borders
    for i in range(len(df_params.columns)):
        cell = table[(0, i)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold', color='white', fontsize=15, 
                           family='Times New Roman')
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # Row styling - Times New Roman, no borders
    for i in range(1, len(table_data)):
        for j in range(len(df_params.columns)):
            # Alternate row colors
            if i == 1:
                cell = table[(i, j)]
                cell.set_facecolor(row1_color)
            else:
                cell = table[(i, j)]
                cell.set_facecolor(row2_color)
            
            # First column (pattern name) bold
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=14, 
                                   family='Times New Roman')
            else:
                cell.set_text_props(fontsize=14, family='Times New Roman')
            
            # Remove borders (or make them subtle)
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
    
    plt.tight_layout(pad=1.2)
    table_img_path = os.path.join(OUTPUT_DIR, "usage_pattern_parameters_table.png")
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.15)
    plt.close()
    print(f"Parameter table image saved to: {table_img_path}")
    
    # Print table
    print("\nWeighted Average Parameters (by time proportion):")
    print("=" * 140)
    print(f"{'Pattern':<15} | {'Brightness':<10} | {'CPU%':<8} | "
          f"{'Freq(MHz)':<10} | {'TP(MB/s)':<10} | {'WiFi':<8} | {'WiFi(dBm)':<10} | {'GPS':<8}")
    print("=" * 140)
    for _, row in df_params.iterrows():
        # Extract values without backslashes
        pattern = str(row['Usage Pattern'])
        brightness = str(row['Brightness\n(0-255)'])
        cpu = str(row['CPU Usage\n(%)'])
        freq = str(row['Frequency\n(MHz)'])
        throughput = str(row['Throughput\n(MB/s)'])
        wifi_status = str(row['WiFi Status\n(0/1)'])
        wifi_intensity = str(row['WiFi Intensity\n(dBm)'])
        gps = str(row['GPS Status\n(0/3)'])
        
        print(f"{pattern:<15} | {brightness:<10} | {cpu:<8} | {freq:<10} | "
              f"{throughput:<10} | {wifi_status:<8} | {wifi_intensity:<10} | {gps:<8}")
    print("=" * 140)

    # ---------- Sensitivity analysis ----------
    print("\n" + "=" * 60)
    print("Sensitivity analysis: each param +/-5%, +/-10%, +/-20%")
    print("=" * 60)
    run_sensitivity_analysis(
        power_model, scenario_row, scenario,
        bp, coef, lam, stats, theta1=theta1, theta2=theta2, t_max_hours=48
    )
    print(f"\nSensitivity figures saved to: {os.path.join(OUTPUT_DIR, 'sensitivity')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
