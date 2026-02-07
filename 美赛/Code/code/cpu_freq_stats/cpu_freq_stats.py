"""
统计 mobile data 中 cpu_usage × frequency_core0~7 的平均值
- 读取所有 data/mobile data 的 *_dynamic_processed.csv
- 计算每行: cpu_usage * mean(frequency_core0,...,frequency_core7)
- 统计整体均值、5%/50%/95% 分位数
- 绘制直方图
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # code -> 项目根目录
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
OUTPUT_DIR = SCRIPT_DIR


def load_all_mobile_data():
    """加载 mobile data 下所有 *_dynamic_processed.csv"""
    pattern = os.path.join(DATA_DIR, "*", "*", "*_dynamic_processed.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到数据文件，路径: {DATA_DIR}")

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            all_dfs.append(df)
        except Exception as e:
            print(f"跳过 {f}: {e}")

    data = pd.concat(all_dfs, ignore_index=True)
    print(f"共加载 {len(files)} 个文件，{len(data)} 行数据")
    return data


def compute_cpu_usage_freq(df):
    """
    计算每行的 cpu_usage × (frequency_core0~7 的平均值)
    返回有效值的一维数组
    """
    freq_cols = [c for c in df.columns if c.startswith("frequency_core")]
    if not freq_cols:
        raise ValueError("未找到 frequency_core 列")

    cpu_usage = pd.to_numeric(df["cpu_usage"], errors="coerce").fillna(0)
    freq_mean = df[freq_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    values = cpu_usage * freq_mean
    # 剔除无效值（NaN、负值等）
    values = values[values.notna() & (values >= 0)]
    return values.values


def main():
    print("加载 mobile data...")
    data = load_all_mobile_data()

    print("计算 cpu_usage × mean(frequency_core0~7)...")
    values = compute_cpu_usage_freq(data)

    # 分位数
    p5 = np.percentile(values, 5)
    p50 = np.percentile(values, 50)
    p95 = np.percentile(values, 95)
    mean_val = np.mean(values)

    print("\n========== 统计结果 ==========")
    print(f"有效样本数: {len(values)}")
    print(f"均值:       {mean_val:.2f}")
    print(f"5% 分位数:  {p5:.2f}")
    print(f"50% 分位数: {p50:.2f}")
    print(f"95% 分位数: {p95:.2f}")

    # 直方图
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(values, bins=80, color="steelblue", edgecolor="white", alpha=0.85)

    # 标注分位线
    for p, label, color in [(p5, "5%", "#e74c3c"), (p50, "50%", "#27ae60"), (p95, "95%", "#9b59b6")]:
        ax.axvline(p, color=color, linestyle="--", linewidth=2, label=f"{label}: {p:.1f}")
    ax.axvline(mean_val, color="orange", linestyle="-", linewidth=2, label=f"均值: {mean_val:.1f}")

    ax.set_xlabel("cpu_usage × mean(frequency_core0~7)")
    ax.set_ylabel("频数")
    ax.set_title("CPU 使用率 × 核心频率均值 分布直方图")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "cpu_freq_histogram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n直方图已保存: {out_path}")


if __name__ == "__main__":
    main()
