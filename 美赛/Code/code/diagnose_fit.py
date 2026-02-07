"""
拟合质量诊断脚本

分析为什么残差看起来很大，以及模型的局限性。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 参数
Q0 = 2.0
r = 0.022
E0 = 3.6

# 读取拟合参数
import json
with open(os.path.join(OUTPUT_DIR, "vocv_fit_params.json"), 'r') as f:
    params = json.load(f)
    k, A, B = params['k'], params['A'], params['B']
    rss = params.get('residual_sum_squares', 0)

print("=" * 80)
print("电池模型拟合质量诊断".center(80))
print("=" * 80)
print()

# 读取数据
DYNAMIC_DATA_PATH = os.path.join(DATA_DIR, "data2", "SP2", "DST_25C", "11_05_2015_SP20-2_DST_50SOC.xls")

def load_data():
    xl = pd.ExcelFile(DYNAMIC_DATA_PATH)
    for sheet in xl.sheet_names:
        if sheet == "Info":
            continue
        df = pd.read_excel(DYNAMIC_DATA_PATH, sheet_name=sheet)
        cur_c = next((c for c in df.columns if "Current" in c and "A)" in c), None)
        volt_c = next((c for c in df.columns if "Voltage" in c and "V)" in c), None)
        dch_c = next((c for c in df.columns if "Discharge" in c and "Capacity" in c), None)
        time_c = next((c for c in df.columns if "Test_Time" in c), None)
        if cur_c and volt_c and dch_c:
            t = df[time_c].values.astype(float)
            I = df[cur_c].values.astype(float)
            U = df[volt_c].values.astype(float)
            Dch = df[dch_c].values.astype(float)
            S = np.clip(1.0 - Dch / Q0, 1e-6, 1.0)
            # 放电段
            mask = I > 0.01
            return t[mask], I[mask], U[mask], S[mask]
    return None, None, None, None

t, I, U_obs, S = load_data()

if t is None:
    print("无法加载数据！")
    exit()

# 降采样
step = max(1, len(t) // 2000)
t = t[::step]
I = I[::step]
U_obs = U_obs[::step]
S = S[::step]

# 计算模型预测
def vocv(S_val):
    S_safe = np.clip(S_val, 1e-6, 1.0)
    return E0 - k / S_safe + A * np.exp(-B * (1 - S_safe) * Q0)

Vocv = vocv(S)
U_model = Vocv - I * r
residuals = U_obs - U_model

# 统计分析
print(f"【数据概况】")
print(f"  数据点数: {len(t)}")
print(f"  时间范围: {t[0]:.1f} - {t[-1]:.1f} s ({(t[-1]-t[0])/3600:.2f} h)")
print(f"  SOC 范围: {S.min():.4f} - {S.max():.4f}")
print(f"  电流范围: {I.min():.2f} - {I.max():.2f} A")
print(f"  电压范围: {U_obs.min():.3f} - {U_obs.max():.3f} V")
print()

print(f"【拟合参数】")
print(f"  E0 = {E0:.4f} V")
print(f"  k = {k:.6f}")
print(f"  A = {A:.6f}")
print(f"  B = {B:.6f}")
print(f"  r = {r:.4f} Ω")
print()

print(f"【误差分析】")
rss_calc = np.sum(residuals**2)
rmse = np.sqrt(rss_calc / len(t))
mae = np.mean(np.abs(residuals))
max_error = np.max(np.abs(residuals))
rel_rmse = rmse / np.mean(U_obs) * 100

print(f"  残差平方和 (RSS) = {rss_calc:.6f}")
print(f"  均方根误差 (RMSE) = {rmse:.6f} V")
print(f"  平均绝对误差 (MAE) = {mae:.6f} V")
print(f"  最大绝对误差 = {max_error:.6f} V")
print(f"  相对误差 (RMSE%) = {rel_rmse:.2f}%")
print()

# 分析残差分布
print(f"【残差分布】")
print(f"  残差均值 = {np.mean(residuals):.6f} V (应接近0)")
print(f"  残差标准差 = {np.std(residuals):.6f} V")
print(f"  残差范围: [{residuals.min():.3f}, {residuals.max():.3f}] V")

# 统计残差在不同范围的比例
within_50mV = np.sum(np.abs(residuals) < 0.05) / len(residuals) * 100
within_100mV = np.sum(np.abs(residuals) < 0.1) / len(residuals) * 100
within_200mV = np.sum(np.abs(residuals) < 0.2) / len(residuals) * 100

print(f"  |残差| < 50mV: {within_50mV:.1f}%")
print(f"  |残差| < 100mV: {within_100mV:.1f}%")
print(f"  |残差| < 200mV: {within_200mV:.1f}%")
print()

print(f"【模型局限性分析】")
print(f"  使用的模型: 一阶等效电路 (U = Vocv(SOC) - I*r)")
print(f"  模型假设:")
print(f"    1. 内阻 r 为常数（实际上 r 随 SOC、温度、老化而变）")
print(f"    2. 忽略了极化效应（电容效应）")
print(f"    3. 忽略了浓差极化")
print(f"    4. Vocv-SOC 使用简化的经验公式")
print()
print(f"  数据特点:")
print(f"    - DST (Dynamic Stress Test) 工况：电流动态变化")
print(f"    - 包含充放电混合场景（虽然只用了放电段）")
print(f"    - 可能存在迟滞效应（充放电OCV不同）")
print()

# 建议
print(f"【拟合质量评价】")
if rel_rmse < 3:
    quality = "优秀"
elif rel_rmse < 5:
    quality = "良好"
elif rel_rmse < 10:
    quality = "可接受"
else:
    quality = "需改进"

print(f"  综合评价: {quality} (相对误差 {rel_rmse:.2f}%)")
print()

print(f"【改进建议】")
print(f"  1. 使用二阶或更高阶等效电路模型（加入RC并联支路）")
print(f"  2. 考虑 SOC 依赖的内阻: r = r0 + r1*exp(-S*k_r)")
print(f"  3. 使用更精确的 Vocv-SOC 关系（如多项式或查表法）")
print(f"  4. 使用恒流放电数据代替 DST 数据（减少动态效应影响）")
print(f"  5. 如果坚持使用当前简单模型，RSS={rss_calc:.1f} 是合理的")
print()
print("=" * 80)

# 绘制诊断图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. 残差直方图
ax = axes[0, 0]
ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零线')
ax.set_xlabel('残差 (V)')
ax.set_ylabel('频数')
ax.set_title(f'残差分布直方图 (均值={np.mean(residuals):.4f}V)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 残差 vs 预测值
ax = axes[0, 1]
ax.scatter(U_model, residuals, alpha=0.5, s=10)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('模型预测电压 (V)')
ax.set_ylabel('残差 (V)')
ax.set_title('残差 vs 预测值 (检查是否存在系统性偏差)')
ax.grid(True, alpha=0.3)

# 3. 残差 vs SOC
ax = axes[1, 0]
ax.scatter(S, residuals, alpha=0.5, s=10, c=I, cmap='viridis')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('SOC')
ax.set_ylabel('残差 (V)')
ax.set_title('残差 vs SOC (颜色表示电流大小)')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('电流 (A)')
ax.grid(True, alpha=0.3)

# 4. Q-Q图（检查正态性）
ax = axes[1, 1]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q图 (检查残差是否服从正态分布)')
ax.grid(True, alpha=0.3)

fig.suptitle('拟合质量诊断图', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fit_diagnostics.png"), dpi=150, bbox_inches='tight')
print(f"诊断图已保存: {OUTPUT_DIR}/fit_diagnostics.png")

plt.show()
