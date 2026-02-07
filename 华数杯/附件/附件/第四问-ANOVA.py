import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 创建DataFrame
data = {
    '编号': range(1, 28),
    '温度': [30, 30, 30, 30, 30, 30, 30, 30, 30, 
            40, 40, 40, 40, 40, 40, 40, 40, 40,
            50, 50, 50, 50, 50, 50, 50, 50, 50],
    '湿度': [50, 50, 50, 70, 70, 70, 90, 90, 90,
            50, 50, 50, 70, 70, 70, 90, 90, 90,
            50, 50, 50, 70, 70, 70, 90, 90, 90],
    '固含量': [6, 6, 6, 8, 8, 8, 10, 10, 10,
             8, 8, 8, 10, 10, 10, 6, 6, 6,
             10, 10, 10, 6, 6, 6, 8, 8, 8],
    '孔面积占比': [18.09, 17.04, 19.45, 9.11, 9.4, 8.85, 9.35, 9.4, 9.08,
                30.33, 31.2, 29.2, 24.12, 24.34, 23.11, 36.01, 35.35, 35.66,
                6.41, 6.62, 6.73, 6.41, 6.62, 6.73, 8.38, 8.3, 8.29]
}

df = pd.DataFrame(data)

# 转换分类变量为类别类型（重要）
df['温度'] = df['温度'].astype('category')
df['湿度'] = df['湿度'].astype('category')
df['固含量'] = df['固含量'].astype('category')

# 1. 描述性统计
print("="*50)
print("描述性统计:")
print(df.groupby('温度')['孔面积占比'].describe())
print("\n")
print(df.groupby('湿度')['孔面积占比'].describe())
print("\n")
print(df.groupby('固含量')['孔面积占比'].describe())

# 2. 三因素方差分析
model = ols('孔面积占比 ~ 温度 + 湿度 + 固含量 + 温度:湿度 + 温度:固含量 + 湿度:固含量 + 温度:湿度:固含量', data=df).fit()
anova_results = anova_lm(model, typ=2)

print("="*50)
print("三因素方差分析结果:")
print(anova_results)

# 3. 事后检验 (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("="*50)
print("温度的事后检验(Tukey HSD):")
tukey_temp = pairwise_tukeyhsd(endog=df['孔面积占比'], groups=df['温度'], alpha=0.05)
print(tukey_temp)

print("\n湿度的事后检验(Tukey HSD):")
tukey_humid = pairwise_tukeyhsd(endog=df['孔面积占比'], groups=df['湿度'], alpha=0.05)
print(tukey_humid)

print("\n固含量的事后检验(Tukey HSD):")
tukey_solid = pairwise_tukeyhsd(endog=df['孔面积占比'], groups=df['固含量'], alpha=0.05)
print(tukey_solid)

# 4. 可视化
plt.figure(figsize=(15, 12))

# 主效应图
plt.subplot(2, 2, 1)
sns.pointplot(x='温度', y='孔面积占比', data=df, ci=95, capsize=0.1)
plt.title('温度的主效应')
plt.xlabel('温度 (°C)')
plt.ylabel('孔面积占比 (%)')

plt.subplot(2, 2, 2)
sns.pointplot(x='湿度', y='孔面积占比', data=df, ci=95, capsize=0.1)
plt.title('湿度的主效应')
plt.xlabel('湿度 (%)')
plt.ylabel('孔面积占比 (%)')

plt.subplot(2, 2, 3)
sns.pointplot(x='固含量', y='孔面积占比', data=df, ci=95, capsize=0.1)
plt.title('固含量的主效应')
plt.xlabel('固含量 (%)')
plt.ylabel('孔面积占比 (%)')

# 交互效应图 (温度 × 固含量)
plt.subplot(2, 2, 4)
sns.pointplot(x='温度', y='孔面积占比', hue='固含量', data=df, ci=95, capsize=0.1, dodge=True)
plt.title('温度 × 固含量交互效应')
plt.xlabel('温度 (°C)')
plt.ylabel('孔面积占比 (%)')
plt.legend(title='固含量(%)')

plt.tight_layout()
plt.savefig('anova_results.png', dpi=300)
plt.show()

# 5. 残差分析
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差 vs. 拟合值')
plt.xlabel('拟合值')
plt.ylabel('残差')

plt.subplot(2, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ图 - 残差正态性检验')

plt.subplot(2, 2, 3)
sns.histplot(residuals, kde=True)
plt.title('残差分布')
plt.xlabel('残差')

plt.subplot(2, 2, 4)
sns.scatterplot(x=range(len(residuals)), y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差序列图')
plt.xlabel('观测序号')
plt.ylabel('残差')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.show()

# 6. 输出最优参数组合
optimal_temp = df.groupby('温度')['孔面积占比'].mean().idxmax()
optimal_humid = df.groupby('湿度')['孔面积占比'].mean().idxmax()
optimal_solid = df.groupby('固含量')['孔面积占比'].mean().idxmax()

# 考虑交互效应后的最优组合
optimal_combo = df.loc[df['孔面积占比'].idxmax()]
print("="*50)
print("最优参数组合:")
print(f"温度: {optimal_combo['温度']}°C")
print(f"湿度: {optimal_combo['湿度']}%")
print(f"固含量: {optimal_combo['固含量']}%")
print(f"预期孔面积占比: {df['孔面积占比'].max():.2f}%")