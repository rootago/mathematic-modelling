"""
按原图描述用关键点重绘「① SOC 对比 (RMSE=3.66%)」：
蓝线阶梯状、红线光滑下降，无尖刺。不依赖从图里采点。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

X_MIN, X_MAX = 0, 800
Y_MIN, Y_MAX = 0, 100


# 实测 SOC：阶梯状，按原图描述的关键点 (时间 min, SOC %)
# 0~50 高平台，100~150 约 90%，250~300 约 75%，450→480 降到 38%，480~520 平台 38%，末尾到 5%
KEY_MEASURED = [
    (0, 97.5), (50, 97), (80, 92), (120, 90), (150, 90),
    (200, 80), (270, 76), (300, 75), (350, 65), (420, 58), (450, 55),
    (480, 38), (520, 38), (580, 28), (650, 15), (800, 5),
]

# 模拟 SOC (含老化)：光滑下降，无尖刺
KEY_SIMULATED = [
    (0, 100), (80, 90), (150, 85), (220, 78), (300, 68), (380, 58), (450, 48),
    (480, 42), (520, 38), (600, 28), (700, 15), (800, 5),
]


def step_interp(t_pts, y_pts, t_out):
    """阶梯插值：在 t_pts 之间保持 y 为左端点的值。"""
    t_out = np.asarray(t_out)
    y_out = np.zeros_like(t_out, dtype=float)
    for i, t in enumerate(t_out):
        # 找最大的 t_pts <= t
        idx = np.searchsorted(t_pts, t, side="right") - 1
        idx = max(0, idx)
        y_out[i] = y_pts[idx]
    return y_out


def smooth_interp(t_pts, y_pts, t_out):
    """线性插值（可改成三次样条更光滑）。"""
    return np.interp(t_out, t_pts, y_pts)


def main():
    t_meas = np.array([p[0] for p in KEY_MEASURED])
    y_meas = np.array([p[1] for p in KEY_MEASURED])
    t_sim = np.array([p[0] for p in KEY_SIMULATED])
    y_sim = np.array([p[1] for p in KEY_SIMULATED])

    t_plot = np.linspace(X_MIN, X_MAX, 801)
    soc_measured = step_interp(t_meas, y_meas, t_plot)
    soc_simulated = smooth_interp(t_sim, y_sim, t_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_plot, soc_measured, "b-", linewidth=2, label="实测 SOC")
    ax.plot(t_plot, soc_simulated, "r--", linewidth=2, label="模拟 SOC (含老化)")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SOC (%)", fontsize=12)
    ax.set_title("① SOC 对比 (RMSE=3.66%)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.35, linestyle="-")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(np.arange(0, X_MAX + 1, 100))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    path_out = "tmp_redrawn.png"
    fig.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {path_out}")


if __name__ == "__main__":
    main()
