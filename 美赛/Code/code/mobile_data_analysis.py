"""
Mobile Data 综合分析 - 符合美赛 O 奖标准的可视化（独立图表）
每个分析指标单独成图，便于论文引用和排版
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置美赛风格的绘图参数
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案（专业期刊风格）
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D00000',
    'neutral': '#6C757D',
    'light': '#E9ECEF'
}

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "mobile data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mobile_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mobile_data(date_folder='20231218', device_id='70a09b5174d07fff', sample_rate=10):
    """加载并预处理 mobile data"""
    file_path = os.path.join(DATA_DIR, date_folder, device_id, 
                             f"{device_id}_{date_folder}_dynamic_processed.csv")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # 采样以减少数据量
    df = df.iloc[::sample_rate, :].reset_index(drop=True)
    
    # 时间转换
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    # 数据清洗
    df['battery_level'] = df['battery_level'].clip(0, 100)
    df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
    df['ram_usage_pct'] = (df['ram_usage'] / (df['ram_usage'] + df['ram_free'])) * 100
    
    print(f"Data loaded: {len(df)} samples (after sampling)")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def plot_battery_level(df):
    """图1: 电池电量与充电状态"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    charging_mask = df['battery_charging_status'] == 3
    ax.plot(df.loc[~charging_mask, 'time_seconds']/3600, 
            df.loc[~charging_mask, 'battery_level'], 
            color=COLORS['primary'], linewidth=2, label='Discharging', alpha=0.9)
    ax.plot(df.loc[charging_mask, 'time_seconds']/3600, 
            df.loc[charging_mask, 'battery_level'], 
            color=COLORS['success'], linewidth=2, label='Charging', alpha=0.9)
    
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Battery Level (%)', fontweight='bold', fontsize=12)
    ax.set_title('Battery Level Over Time', fontweight='bold', fontsize=14, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '01_battery_level.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_battery_current_power(df):
    """图2: 电池电流与功率"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['time_seconds']/3600, df['battery_current'], 
                     color=COLORS['secondary'], linewidth=1.5, label='Current', alpha=0.8)
    line2 = ax2.plot(df['time_seconds']/3600, df['battery_power'], 
                     color=COLORS['accent'], linewidth=1.5, label='Power', alpha=0.8)
    
    ax1.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Battery Current (mA)', fontweight='bold', fontsize=12, color=COLORS['secondary'])
    ax2.set_ylabel('Battery Power (W)', fontweight='bold', fontsize=12, color=COLORS['accent'])
    ax1.set_title('Battery Current and Power Consumption', fontweight='bold', fontsize=14, pad=15)
    
    ax1.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '02_battery_current_power.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_battery_temperature(df):
    """图3: 电池温度分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    temp_data = df['battery_temperature'].dropna()
    n, bins, patches = ax.hist(temp_data, bins=40, color=COLORS['accent'], 
                                alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # 渐变色效果
    cm = plt.cm.YlOrRd
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax.axvline(temp_data.mean(), color=COLORS['warning'], linestyle='--', 
               linewidth=2.5, label=f'Mean: {temp_data.mean():.1f}°C')
    ax.axvline(temp_data.median(), color=COLORS['success'], linestyle='--', 
               linewidth=2.5, label=f'Median: {temp_data.median():.1f}°C')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Battery Temperature Distribution', fontweight='bold', fontsize=14, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '03_battery_temperature.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_battery_drain_rate(df):
    """图4: 电池消耗速率（按屏幕状态）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_discharge = df[df['battery_charging_status'] != 3].copy()
    
    if len(df_discharge) > 10:
        df_discharge['battery_rate'] = -df_discharge['battery_level'].diff() / \
                                        (df_discharge['time_seconds'].diff() / 3600)
        df_discharge['battery_rate'] = df_discharge['battery_rate'].clip(-50, 50)
        
        screen_on = df_discharge[df_discharge['screen_status'] == 1]['battery_rate'].dropna()
        screen_off = df_discharge[df_discharge['screen_status'] == 0]['battery_rate'].dropna()
        
        positions = [1, 2]
        data_to_plot = [screen_on, screen_off]
        
        bp = ax.boxplot(data_to_plot, positions=positions, labels=['Screen On', 'Screen Off'],
                       patch_artist=True, widths=0.5, showfliers=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkred'))
        
        colors = [COLORS['primary'], COLORS['neutral']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Battery Drain Rate (%/hour)', fontweight='bold', fontsize=12)
        ax.set_title('Battery Consumption Rate by Screen Status', fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0, max(20, df_discharge['battery_rate'].quantile(0.95))])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '04_battery_drain_rate.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_screen_usage_timeline(df):
    """图5: 屏幕使用时间线 - 使用水平条状图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 找出屏幕开启的连续时间段
    df_sorted = df.sort_values('time_seconds').reset_index(drop=True)
    screen_on_periods = []
    
    in_session = False
    start_time = None
    
    for idx, row in df_sorted.iterrows():
        if row['screen_status'] == 1 and not in_session:
            # 开始一个新的屏幕开启时段
            start_time = row['time_seconds'] / 3600
            in_session = True
        elif row['screen_status'] == 0 and in_session:
            # 结束当前屏幕开启时段
            end_time = row['time_seconds'] / 3600
            screen_on_periods.append((start_time, end_time - start_time))
            in_session = False
    
    # 如果最后还在开启状态，添加最后一段
    if in_session:
        end_time = df_sorted.iloc[-1]['time_seconds'] / 3600
        screen_on_periods.append((start_time, end_time - start_time))
    
    # 绘制水平条状图
    for start, duration in screen_on_periods:
        ax.barh(1, duration, left=start, height=0.8, 
                color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加网格和标签
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Screen Status', fontweight='bold', fontsize=12)
    ax.set_title('Screen Usage Timeline (Blue bars indicate screen-on periods)', 
                fontweight='bold', fontsize=14, pad=15)
    ax.set_ylim([0.5, 1.5])
    ax.set_yticks([1])
    ax.set_yticklabels(['Screen'])
    ax.set_xlim([0, df_sorted['time_seconds'].max() / 3600])
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # 添加统计信息
    total_hours = df_sorted['time_seconds'].max() / 3600
    screen_on_hours = df_sorted[df_sorted['screen_status'] == 1].shape[0] * 10 / 3600  # 每条记录10秒
    usage_pct = (screen_on_hours / total_hours) * 100
    
    ax.text(0.02, 0.95, f'Total Duration: {total_hours:.1f} hours\nScreen On: {screen_on_hours:.2f} hours ({usage_pct:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '05_screen_timeline.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_hourly_screen_usage(df):
    """图6: 每小时屏幕使用时长和电池消耗"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # 计算每小时的屏幕使用时长（分钟）
    # 每条记录代表10秒，screen_status=1 表示屏幕开启
    hourly_stats = df.groupby('hour').agg({
        'screen_status': 'sum',  # 开启的记录数
        'battery_level': lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 1 else 0  # 电量变化
    })
    
    # 转换为分钟（每条记录10秒）
    hourly_stats['usage_minutes'] = hourly_stats['screen_status'] * 10 / 60
    hourly_stats['battery_drain'] = hourly_stats['battery_level'].clip(lower=0)  # 只保留正值（消耗）
    
    # 绘制柱状图
    x = hourly_stats.index
    width = 0.8
    
    bars1 = ax1.bar(x, hourly_stats['usage_minutes'], width=width,
                    color=COLORS['primary'], alpha=0.7, edgecolor='black', 
                    linewidth=1.2, label='Screen Usage (minutes)')
    
    # 渐变色效果
    max_usage = hourly_stats['usage_minutes'].max()
    if max_usage > 0:
        for i, bar in enumerate(bars1):
            alpha_val = 0.4 + 0.6 * (hourly_stats['usage_minutes'].iloc[i] / max_usage)
            bar.set_alpha(np.clip(alpha_val, 0.3, 1.0))
    
    # 绘制电池消耗曲线
    line2 = ax2.plot(x, hourly_stats['battery_drain'], 
                     color=COLORS['warning'], linewidth=3, marker='o', 
                     markersize=8, label='Battery Drain (%)', alpha=0.9)
    
    # 设置标签
    ax1.set_xlabel('Hour of Day', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Screen Usage Time (minutes)', fontweight='bold', fontsize=12, 
                   color=COLORS['primary'])
    ax2.set_ylabel('Battery Drain (%)', fontweight='bold', fontsize=12, 
                   color=COLORS['warning'])
    
    ax1.set_title('Hourly Screen Usage and Battery Consumption Pattern', 
                 fontweight='bold', fontsize=14, pad=15)
    
    ax1.set_xticks(range(0, 24, 2))
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['warning'])
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0, max(60, hourly_stats['usage_minutes'].max() * 1.1)])
    ax2.set_ylim([0, max(10, hourly_stats['battery_drain'].max() * 1.2)])
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 添加峰值标注
    peak_hour = hourly_stats['usage_minutes'].idxmax()
    peak_value = hourly_stats['usage_minutes'].max()
    if peak_value > 0:
        ax1.annotate(f'Peak: {peak_value:.1f} min\nat {peak_hour}:00',
                    xy=(peak_hour, peak_value), 
                    xytext=(peak_hour + 1, peak_value * 0.8),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '06_hourly_screen_usage.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_cpu_by_screen(df):
    """图7: CPU使用率分布（按屏幕状态）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cpu_screen_on = df[df['screen_status'] == 1]['cpu_usage'].dropna()
    cpu_screen_off = df[df['screen_status'] == 0]['cpu_usage'].dropna()
    
    if len(cpu_screen_on) > 5 and len(cpu_screen_off) > 5:
        parts = ax.violinplot([cpu_screen_on, cpu_screen_off], 
                              positions=[1, 2], widths=0.7,
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([COLORS['primary'], COLORS['neutral']][i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1.5)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Screen On', 'Screen Off'])
    else:
        data_to_plot = []
        labels = []
        if len(cpu_screen_on) > 0:
            data_to_plot.append(cpu_screen_on)
            labels.append('Screen On')
        if len(cpu_screen_off) > 0:
            data_to_plot.append(cpu_screen_off)
            labels.append('Screen Off')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], [COLORS['primary'], COLORS['neutral']][:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    
    ax.set_ylabel('CPU Usage (%)', fontweight='bold', fontsize=12)
    ax.set_title('CPU Usage Distribution by Screen Status', fontweight='bold', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '07_cpu_by_screen.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_network_traffic(df):
    """图8: 网络流量速率（更直观）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算流量变化率（KB/s）
    df['wifi_total'] = df['wifi_rx'] + df['wifi_tx']
    df['mobile_total'] = df['mobile_rx'] + df['mobile_tx']
    
    # 计算速率（每10秒的变化，转换为 KB/s）
    df['wifi_rate'] = df['wifi_total'].diff() / 10 / 1024  # KB/s
    df['mobile_rate'] = df['mobile_total'].diff() / 10 / 1024  # KB/s
    
    # 平滑处理（移动平均）
    window = 6  # 1分钟窗口
    df['wifi_rate_smooth'] = df['wifi_rate'].rolling(window=window, min_periods=1).mean()
    df['mobile_rate_smooth'] = df['mobile_rate'].rolling(window=window, min_periods=1).mean()
    
    # 限制异常值
    df['wifi_rate_smooth'] = df['wifi_rate_smooth'].clip(0, df['wifi_rate_smooth'].quantile(0.99))
    df['mobile_rate_smooth'] = df['mobile_rate_smooth'].clip(0, df['mobile_rate_smooth'].quantile(0.99))
    
    # 绘图
    ax.fill_between(df['time_seconds']/3600, 0, df['wifi_rate_smooth'], 
                    color=COLORS['primary'], alpha=0.5, label='WiFi')
    ax.plot(df['time_seconds']/3600, df['wifi_rate_smooth'], 
            color=COLORS['primary'], linewidth=2, alpha=0.9)
    
    ax.fill_between(df['time_seconds']/3600, 0, df['mobile_rate_smooth'], 
                    color=COLORS['secondary'], alpha=0.5, label='Mobile Data')
    ax.plot(df['time_seconds']/3600, df['mobile_rate_smooth'], 
            color=COLORS['secondary'], linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Network Traffic Rate (KB/s)', fontweight='bold', fontsize=12)
    ax.set_title('Network Traffic Rate Over Time (1-min smoothed)', 
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # 添加统计信息
    total_wifi_mb = df['wifi_total'].max() / 1024 / 1024
    total_mobile_mb = df['mobile_total'].max() / 1024 / 1024
    ax.text(0.02, 0.98, 
           f'Total WiFi: {total_wifi_mb:.1f} MB\nTotal Mobile: {total_mobile_mb:.1f} MB',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '08_network_traffic.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_wifi_signal(df):
    """图9: WiFi信号强度分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wifi_connected = df[df['wifi_status'] == 1]['wifi_intensity'].dropna()
    
    if len(wifi_connected) > 0:
        n, bins, patches = ax.hist(wifi_connected, bins=35, color=COLORS['success'], 
                                    alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # 渐变色
        cm = plt.cm.Greens
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(0.4 + 0.6 * c))
        
        ax.axvline(wifi_connected.mean(), color=COLORS['warning'], 
                  linestyle='--', linewidth=2.5, 
                  label=f'Mean: {wifi_connected.mean():.1f} dBm')
        ax.axvline(wifi_connected.median(), color='darkblue', 
                  linestyle='--', linewidth=2.5, 
                  label=f'Median: {wifi_connected.median():.1f} dBm')
        
        ax.set_xlabel('WiFi Signal Strength (dBm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title('WiFi Signal Strength Distribution', fontweight='bold', fontsize=14, pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '09_wifi_signal.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_network_type_pie(df):
    """图10: 网络连接类型分布"""
    fig, ax = plt.subplots(figsize=(9, 9))
    
    wifi_on_count = (df['wifi_status'] == 1).sum()
    mobile_on_count = (df['mobile_status'] > 0).sum()
    both_off_count = ((df['wifi_status'] == 0) & (df['mobile_status'] == 0)).sum()
    
    sizes = [wifi_on_count, mobile_on_count, both_off_count]
    labels = ['WiFi Only', 'Mobile Data', 'Offline']
    colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['neutral']]
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=explode, shadow=True,
                                       textprops={'fontweight': 'bold', 'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    
    ax.set_title('Network Connection Type Distribution', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '10_network_type.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_wifi_speed(df):
    """图11: WiFi连接质量分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    wifi_data = df[df['wifi_status'] == 1][['time_seconds', 'wifi_speed', 'wifi_intensity']].dropna()
    
    if len(wifi_data) > 10:
        # 左图：WiFi速度时间序列
        ax1.plot(wifi_data['time_seconds']/3600, wifi_data['wifi_speed'],
                color=COLORS['primary'], linewidth=1.5, alpha=0.7)
        ax1.fill_between(wifi_data['time_seconds']/3600, 0, wifi_data['wifi_speed'],
                        color=COLORS['primary'], alpha=0.3)
        
        # 添加平均线
        mean_speed = wifi_data['wifi_speed'].mean()
        ax1.axhline(mean_speed, color=COLORS['warning'], linestyle='--', 
                   linewidth=2, label=f'Average: {mean_speed:.1f} Mbps')
        
        ax1.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('WiFi Speed (Mbps)', fontweight='bold', fontsize=12)
        ax1.set_title('WiFi Connection Speed Over Time', fontweight='bold', fontsize=12, pad=10)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        
        # 右图：速度 vs 信号强度散点图
        scatter = ax2.scatter(wifi_data['wifi_intensity'], wifi_data['wifi_speed'],
                            c=wifi_data['wifi_speed'], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 添加趋势线
        if len(wifi_data) > 20:
            z = np.polyfit(wifi_data['wifi_intensity'], wifi_data['wifi_speed'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(wifi_data['wifi_intensity'].min(), 
                                wifi_data['wifi_intensity'].max(), 100)
            ax2.plot(x_trend, p(x_trend), color=COLORS['warning'], 
                    linestyle='--', linewidth=2.5, 
                    label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
        
        ax2.set_xlabel('Signal Strength (dBm)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Connection Speed (Mbps)', fontweight='bold', fontsize=12)
        ax2.set_title('Speed vs Signal Strength', fontweight='bold', fontsize=12, pad=10)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label('Speed (Mbps)', fontweight='bold', fontsize=10)
    else:
        # 如果数据不足，显示提示
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'Insufficient WiFi data', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.suptitle('WiFi Connection Quality Analysis', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '11_wifi_speed.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_cpu_usage(df):
    """图12: CPU使用率时间序列"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['time_seconds']/3600, df['cpu_usage'], 
            color=COLORS['accent'], linewidth=1.5, alpha=0.8)
    ax.fill_between(df['time_seconds']/3600, 0, df['cpu_usage'], 
                    color=COLORS['accent'], alpha=0.3)
    
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax.set_ylabel('CPU Usage (%)', fontweight='bold', fontsize=12)
    ax.set_title('CPU Usage Over Time', fontweight='bold', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '12_cpu_usage.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_ram_usage(df):
    """图13: 内存使用率"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['time_seconds']/3600, df['ram_usage_pct'], 
            color=COLORS['primary'], linewidth=2, alpha=0.9)
    ax.fill_between(df['time_seconds']/3600, 0, df['ram_usage_pct'], 
                    color=COLORS['primary'], alpha=0.2)
    
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax.set_ylabel('RAM Usage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Memory Usage Over Time', fontweight='bold', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '13_ram_usage.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_cpu_frequency_heatmap(df):
    """图14: CPU核心频率和平均频率分析"""
    freq_cols = [f'frequency_core{i}' for i in range(8) if f'frequency_core{i}' in df.columns]
    
    if len(freq_cols) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：热力图
        sample_indices = np.linspace(0, len(df)-1, min(300, len(df)), dtype=int)
        freq_data = df.loc[sample_indices, freq_cols].T.values
        time_sample = df.loc[sample_indices, 'time_seconds'].values / 3600
        
        im = ax1.imshow(freq_data, aspect='auto', cmap='RdYlGn_r', 
                       interpolation='bilinear', 
                       extent=[time_sample[0], time_sample[-1], 0, len(freq_cols)])
        
        ax1.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('CPU Core', fontweight='bold', fontsize=12)
        ax1.set_title('CPU Core Frequency Heatmap', fontweight='bold', fontsize=12, pad=10)
        ax1.set_yticks(np.arange(len(freq_cols)) + 0.5)
        ax1.set_yticklabels([f'Core {i}' for i in range(len(freq_cols))])
        
        cbar = plt.colorbar(im, ax=ax1, pad=0.02)
        cbar.set_label('Frequency (MHz)', fontweight='bold', fontsize=11)
        
        # 下图：平均频率时间序列
        df['avg_frequency'] = df[freq_cols].mean(axis=1)
        df['max_frequency'] = df[freq_cols].max(axis=1)
        
        ax2.plot(df['time_seconds']/3600, df['avg_frequency'], 
                color=COLORS['primary'], linewidth=2, label='Average Frequency', alpha=0.8)
        ax2.fill_between(df['time_seconds']/3600, 0, df['avg_frequency'], 
                        color=COLORS['primary'], alpha=0.3)
        ax2.plot(df['time_seconds']/3600, df['max_frequency'], 
                color=COLORS['warning'], linewidth=1.5, linestyle='--', 
                label='Max Frequency', alpha=0.7)
        
        # 添加平均线
        mean_freq = df['avg_frequency'].mean()
        ax2.axhline(mean_freq, color=COLORS['secondary'], linestyle=':', 
                   linewidth=2, label=f'Overall Avg: {mean_freq:.0f} MHz')
        
        ax2.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Frequency (MHz)', fontweight='bold', fontsize=12)
        ax2.set_title('Average CPU Frequency Over Time', fontweight='bold', fontsize=12, pad=10)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
        
        plt.suptitle('CPU Multi-Core Frequency Analysis', fontweight='bold', fontsize=14, y=0.995)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'CPU frequency data not available', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '14_cpu_frequency.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_storage_usage(df):
    """图15: 存储使用统计"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    storage_data = {
        'RAM Used': df['ram_usage'].mean() / 1024**3,
        'RAM Free': df['ram_free'].mean() / 1024**3,
        'ROM Used': df['rom_usage'].mean() / 1024**3,
        'ROM Free': df['rom_free'].mean() / 1024**3
    }
    
    categories = list(storage_data.keys())
    values = list(storage_data.values())
    colors_bar = [COLORS['warning'], COLORS['success'], COLORS['warning'], COLORS['success']]
    
    bars = ax.barh(categories, values, color=colors_bar, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.2, i, f'{val:.2f} GB', 
               va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Storage (GB)', fontweight='bold', fontsize=12)
    ax.set_title('Average Storage Usage', fontweight='bold', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '15_storage_usage.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_correlation_matrix(df):
    """图16: 功耗影响因素相关性热力图"""
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # 计算网络活动强度（WiFi + 移动网络流量速率）
    df['network_activity'] = (df['wifi_rx'] + df['wifi_tx'] + df['mobile_rx'] + df['mobile_tx']).diff().fillna(0)
    df['network_activity'] = df['network_activity'].clip(0, df['network_activity'].quantile(0.99))
    
    # 选择与功耗相关的关键变量
    # 1. 屏幕亮度 (bright_level)
    # 2. 处理器负载 (cpu_usage)
    # 3. 网络活动 (network_activity)
    # 4. 环境温度/电池温度 (battery_temperature)
    # 5. 电池电流/功耗 (battery_current)
    # 6. 电池电量 (battery_level)
    # 7. 屏幕状态 (screen_status)
    # 8. RAM使用率 (ram_usage_pct)
    
    key_vars = ['bright_level', 'cpu_usage', 'network_activity', 'battery_temperature',
                'battery_current', 'battery_level', 'screen_status', 'ram_usage_pct']
    
    # 检查是否有 bright_level 列
    if 'bright_level' not in df.columns or df['bright_level'].isna().all():
        # 如果没有亮度数据，使用 screen_on_time 替代
        key_vars[0] = 'screen_on_time'
        labels_short = ['Screen\nTime', 'CPU\nLoad', 'Network\nActivity', 'Battery\nTemp',
                       'Power\nConsumption', 'Battery\nLevel', 'Screen\nStatus', 'RAM\nUsage']
        labels_long = ['Screen Time', 'CPU Load', 'Network Activity', 'Battery Temp',
                      'Power Consumption', 'Battery Level', 'Screen Status', 'RAM Usage']
    else:
        labels_short = ['Screen\nBrightness', 'CPU\nLoad', 'Network\nActivity', 'Battery\nTemp',
                       'Power\nConsumption', 'Battery\nLevel', 'Screen\nStatus', 'RAM\nUsage']
        labels_long = ['Screen Brightness', 'CPU Load', 'Network Activity', 'Battery Temp',
                      'Power Consumption', 'Battery Level', 'Screen Status', 'RAM Usage']
    
    df_corr = df[key_vars].dropna()
    
    if len(df_corr) > 10:
        corr_matrix = df_corr.corr()
        
        # 使用更专业的热力图
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(key_vars)))
        ax.set_yticks(range(len(key_vars)))
        ax.set_xticklabels(labels_short, rotation=0, ha='center', fontsize=11, fontweight='bold')
        ax.set_yticklabels(labels_long, fontsize=11, fontweight='bold')
        
        # 添加数值标签和边框
        for i in range(len(key_vars)):
            for j in range(len(key_vars)):
                # 根据相关性强度选择文字颜色
                text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color=text_color, 
                             fontweight='bold', fontsize=11)
        
        # 添加网格线
        for i in range(len(key_vars) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2)
            ax.axvline(i - 0.5, color='white', linewidth=2)
        
        ax.set_title('Power Consumption Factors Correlation Analysis\n' + 
                    '(Screen, Processor, Network, Temperature, Background Apps)', 
                    fontweight='bold', fontsize=14, pad=20)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # 使用 tight_layout 并预留底部空间
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    # 添加文本说明（调整位置避免重叠）
    fig.text(0.5, 0.02, 
            'Note: Power consumption is affected by screen brightness, processor load, network activity,\n' +
            'battery temperature, and background applications. Higher correlation values indicate stronger relationships.',
            ha='center', fontsize=9, style='italic', wrap=True)
    
    output_path = os.path.join(OUTPUT_DIR, '16_correlation_matrix.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_cpu_vs_battery(df):
    """图17: CPU使用率 vs 功耗关系分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    discharge_data = df[df['battery_charging_status'] != 3][['cpu_usage', 'battery_current', 
                                                               'battery_temperature', 'screen_status']].dropna()
    
    if len(discharge_data) > 10:
        # 左图：CPU vs 电流散点图（按温度着色）
        scatter1 = ax1.scatter(discharge_data['cpu_usage'], 
                              -discharge_data['battery_current'],
                              c=discharge_data['battery_temperature'],
                              cmap='YlOrRd', alpha=0.5, s=40, 
                              edgecolors='none')
        
        # 趋势线
        z = np.polyfit(discharge_data['cpu_usage'], -discharge_data['battery_current'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(discharge_data['cpu_usage'].min(), 
                            discharge_data['cpu_usage'].max(), 100)
        ax1.plot(x_trend, p(x_trend), color=COLORS['warning'], 
                linestyle='--', linewidth=3, 
                label=f'y = {z[0]:.2f}x + {z[1]:.1f}')
        
        # 计算相关系数
        corr = discharge_data[['cpu_usage', 'battery_current']].corr().iloc[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('CPU Usage (%)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Battery Current (mA)', fontweight='bold', fontsize=12)
        ax1.set_title('CPU Load vs Power Consumption', fontweight='bold', fontsize=12, pad=10)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02)
        cbar1.set_label('Temp (°C)', fontweight='bold', fontsize=10)
        
        # 右图：按屏幕状态分组的箱线图
        screen_on_data = discharge_data[discharge_data['screen_status'] == 1]
        screen_off_data = discharge_data[discharge_data['screen_status'] == 0]
        
        data_to_plot = []
        labels = []
        colors = []
        
        if len(screen_on_data) > 0:
            data_to_plot.append(-screen_on_data['battery_current'])
            labels.append('Screen On')
            colors.append(COLORS['primary'])
        
        if len(screen_off_data) > 0:
            data_to_plot.append(-screen_off_data['battery_current'])
            labels.append('Screen Off')
            colors.append(COLORS['neutral'])
        
        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                            widths=0.5, showfliers=True,
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            medianprops=dict(linewidth=2.5, color='darkred'))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 添加均值标注
            for i, data in enumerate(data_to_plot):
                mean_val = data.mean()
                ax2.plot(i + 1, mean_val, marker='D', color='red', 
                        markersize=10, label='Mean' if i == 0 else '')
                ax2.text(i + 1, mean_val, f' {mean_val:.1f}', 
                        fontsize=10, fontweight='bold', va='bottom')
        
        ax2.set_ylabel('Battery Current (mA)', fontweight='bold', fontsize=12)
        ax2.set_title('Power Consumption by Screen Status', fontweight='bold', fontsize=12, pad=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    else:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'Insufficient discharge data', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.suptitle('CPU-Battery Power Consumption Analysis', fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '17_cpu_vs_battery.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def plot_screen_impact(df):
    """图18: 屏幕状态对系统资源的影响"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'ram_usage_pct' in df.columns and len(df['screen_status'].unique()) > 0:
        screen_stats = df.groupby('screen_status').agg({
            'battery_current': lambda x: -x.mean() if len(x) > 0 else 0,
            'cpu_usage': 'mean',
            'ram_usage_pct': 'mean'
        })
        
        if len(screen_stats) > 0:
            x = np.arange(len(screen_stats))
            width = 0.25
            
            bars1 = ax.bar(x - width, screen_stats['battery_current'], width, 
                          label='Battery Current (mA)', color=COLORS['warning'], 
                          alpha=0.8, edgecolor='black', linewidth=1.2)
            bars2 = ax.bar(x, screen_stats['cpu_usage'], width, 
                          label='CPU Usage (%)', color=COLORS['accent'], 
                          alpha=0.8, edgecolor='black', linewidth=1.2)
            bars3 = ax.bar(x + width, screen_stats['ram_usage_pct'], width, 
                          label='RAM Usage (%)', color=COLORS['primary'], 
                          alpha=0.8, edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel('Screen Status', fontweight='bold', fontsize=12)
            ax.set_ylabel('Value', fontweight='bold', fontsize=12)
            ax.set_title('System Resource Impact by Screen Status', 
                        fontweight='bold', fontsize=14, pad=15)
            ax.set_xticks(x)
            labels = ['Screen Off' if idx == 0 else 'Screen On' for idx in screen_stats.index]
            ax.set_xticklabels(labels)
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper left')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '18_screen_impact.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """主函数：执行完整分析流程"""
    print("=" * 70)
    print("Mobile Data Comprehensive Analysis - Individual Plots")
    print("=" * 70)
    
    # 加载数据
    df = load_mobile_data(date_folder='20231218', device_id='70a09b5174d07fff', sample_rate=10)
    
    print("\n" + "=" * 70)
    print("Generating Individual Visualizations...")
    print("=" * 70)
    
    # 生成各类分析图
    plot_functions = [
        ("Battery Level", plot_battery_level),
        ("Battery Current & Power", plot_battery_current_power),
        ("Battery Temperature", plot_battery_temperature),
        ("Battery Drain Rate", plot_battery_drain_rate),
        ("Screen Timeline", plot_screen_usage_timeline),
        ("Hourly Screen Usage", plot_hourly_screen_usage),
        ("CPU by Screen", plot_cpu_by_screen),
        ("Network Traffic", plot_network_traffic),
        ("WiFi Signal", plot_wifi_signal),
        ("Network Type", plot_network_type_pie),
        ("WiFi Speed", plot_wifi_speed),
        ("CPU Usage", plot_cpu_usage),
        ("RAM Usage", plot_ram_usage),
        ("CPU Frequency", plot_cpu_frequency_heatmap),
        ("Storage Usage", plot_storage_usage),
        ("Correlation Matrix", plot_correlation_matrix),
        ("CPU vs Battery", plot_cpu_vs_battery),
        ("Screen Impact", plot_screen_impact),
    ]
    
    for i, (name, func) in enumerate(plot_functions, 1):
        print(f"\n[{i}/{len(plot_functions)}] {name}...")
        func(df)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"All {len(plot_functions)} visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 统计信息
    print("\n" + "=" * 70)
    print("Key Statistics Summary:")
    print("=" * 70)
    print(f"Total duration: {df['time_seconds'].max()/3600:.2f} hours")
    print(f"Battery level range: {df['battery_level'].min():.0f}% - {df['battery_level'].max():.0f}%")
    print(f"Average CPU usage: {df['cpu_usage'].mean():.1f}%")
    print(f"Screen on time: {(df['screen_status'].sum() * 10 / 3600):.2f} hours")
    print(f"Average battery temperature: {df['battery_temperature'].mean():.1f}°C")
    print(f"Average RAM usage: {df['ram_usage_pct'].mean():.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
