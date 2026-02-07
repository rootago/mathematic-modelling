import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取Excel数据
file_path = '附件2.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=9, usecols='A:E', nrows=29)

# 清理列名
df.columns = ['编号', '温度 （°C）', '湿度(%)', '固含量(%)', '孔面积占比(%)']
df = df.drop(0,axis=0)
df = df.drop(1,axis=0)
df.index = np.arange(1,28)
pass
# 2. 计算Pearson相关系数矩阵
corr_matrix = df[['温度 （°C）', '湿度(%)', '固含量(%)', '孔面积占比(%)']].corr(method='pearson')
# df['温度 （°C）'] = df['温度 （°C）'] + 273.15
# 3. 创建热力图
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角遮罩
sns.heatmap(corr_matrix, 
            # mask=mask,  # 只显示下三角
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1, 
            vmax=1,
            linewidths=0.2,
            cbar_kws={'label': 'Pearson相关系数'})

# 设置标题和标签
plt.title('多孔膜参数与孔面积占比的Pearson相关系数热力图', fontsize=20, pad=10)
plt.xticks(rotation=10, ha='right', fontsize=16)
plt.yticks(fontsize=16)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('pearson_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 打印相关系数矩阵
print("Pearson相关系数矩阵:")
print(corr_matrix)