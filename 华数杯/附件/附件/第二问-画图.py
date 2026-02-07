import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

colors = [
    (0.0, '#2a2a72'), 
    (0.2, '#009ffd'), 
    (0.4, '#00c9a7'),   
    (0.6, '#ffd166'),   
    (0.8, '#ff6b6b'),   
    (1.0, '#ef476f')    
]
cmap = LinearSegmentedColormap.from_list('custom_heat', colors)

results_df = pd.read_excel("孔隙率计算结果1.xlsx")

T_values = np.sort(results_df['温度(℃)'].unique())
H_values = np.sort(results_df['湿度(%)'].unique())
SC_values = np.sort(results_df['SC'].unique())

T_grid, H_grid, SC_grid = np.meshgrid(T_values, H_values, SC_values, indexing='ij')

Pp_grid = np.zeros_like(T_grid, dtype=float)
for i, T in enumerate(T_values):
    for j, H in enumerate(H_values):
        for k, SC in enumerate(SC_values):
            mask = (results_df['温度(℃)'] == T) & (results_df['湿度(%)'] == H) & (results_df['SC'] == SC)
            if mask.any():
                Pp_grid[i, j, k] = results_df.loc[mask, '孔隙率(Pp)'].values[0]
            else:
                Pp_grid[i, j, k] = np.nan

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

scatter = ax1.scatter(
    T_grid.flatten(), 
    H_grid.flatten(), 
    SC_grid.flatten() * 100,
    c=Pp_grid.flatten(), 
    cmap=cmap,
    s=50,
    alpha=0.7,
    edgecolor='w',
    linewidth=0.3
)

cbar = fig1.colorbar(scatter, ax=ax1, pad=0.1,shrink = 0.5)
cbar.set_label('Porosity (Pp)', fontsize=12)

ax1.set_xlabel('Temperature (°C)', fontsize=20, labelpad=10)
ax1.set_ylabel('Humidity (%)', fontsize=20, labelpad=10)
ax1.set_zlabel('SC (%)', fontsize=20, labelpad=10)
ax1.set_title('Porosity Distribution', fontsize=20,pad=5)

ax1.grid(True, linestyle='--', alpha=0.5)
ax1.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.savefig('porosity_3d_main.png', dpi=300, bbox_inches='tight')
plt.show()
