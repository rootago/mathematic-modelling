"""
MATLAB数据转Excel转换器
支持将NASA电池数据集的.mat文件转换为Excel格式
"""

import scipy.io as sio
import pandas as pd
import numpy as np
import os
from pathlib import Path


def extract_battery_data(mat_file_path):
    """
    从.mat文件中提取电池数据
    
    参数:
        mat_file_path: .mat文件路径
    
    返回:
        包含不同数据类型的字典
    """
    # 读取.mat文件
    mat_data = sio.loadmat(mat_file_path)
    
    # 获取电池名称（通常是文件名，如B0005）
    battery_name = os.path.basename(mat_file_path).replace('.mat', '')
    
    # 提取数据结构
    # NASA数据集通常包含 'cycle' 字段
    if battery_name in mat_data:
        battery_data = mat_data[battery_name]
    else:
        # 尝试其他可能的键
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if keys:
            battery_data = mat_data[keys[0]]
        else:
            print(f"警告: 无法找到数据键在文件 {mat_file_path}")
            return None
    
    result = {
        'battery_name': battery_name,
        'charge_data': [],
        'discharge_data': [],
        'impedance_data': []
    }
    
    try:
        # 处理循环数据
        if battery_data.dtype.names and 'cycle' in battery_data.dtype.names:
            cycles = battery_data['cycle'][0, 0]
            
            for i in range(len(cycles[0])):
                cycle = cycles[0][i]
                
                # 获取循环类型
                cycle_type = str(cycle['type'][0]) if 'type' in cycle.dtype.names else ''
                ambient_temp = float(cycle['ambient_temperature'][0][0]) if 'ambient_temperature' in cycle.dtype.names else np.nan
                
                # 获取数据部分
                if 'data' in cycle.dtype.names:
                    data = cycle['data'][0, 0]
                    
                    cycle_info = {
                        'cycle_number': i + 1,
                        'type': cycle_type,
                        'ambient_temperature': ambient_temp,
                    }
                    
                    # 根据类型提取不同字段
                    if 'charge' in cycle_type.lower():
                        if 'Voltage_measured' in data.dtype.names:
                            cycle_info['voltage'] = data['Voltage_measured'][0].flatten()
                        if 'Current_measured' in data.dtype.names:
                            cycle_info['current'] = data['Current_measured'][0].flatten()
                        if 'Temperature_measured' in data.dtype.names:
                            cycle_info['temperature'] = data['Temperature_measured'][0].flatten()
                        if 'Time' in data.dtype.names:
                            cycle_info['time'] = data['Time'][0].flatten()
                        if 'Voltage_charge' in data.dtype.names:
                            cycle_info['voltage_charge'] = data['Voltage_charge'][0].flatten()
                        if 'Current_charge' in data.dtype.names:
                            cycle_info['current_charge'] = data['Current_charge'][0].flatten()
                        
                        result['charge_data'].append(cycle_info)
                    
                    elif 'discharge' in cycle_type.lower():
                        if 'Voltage_measured' in data.dtype.names:
                            cycle_info['voltage'] = data['Voltage_measured'][0].flatten()
                        if 'Current_measured' in data.dtype.names:
                            cycle_info['current'] = data['Current_measured'][0].flatten()
                        if 'Temperature_measured' in data.dtype.names:
                            cycle_info['temperature'] = data['Temperature_measured'][0].flatten()
                        if 'Time' in data.dtype.names:
                            cycle_info['time'] = data['Time'][0].flatten()
                        if 'Capacity' in data.dtype.names:
                            cycle_info['capacity'] = data['Capacity'][0].flatten()
                        
                        result['discharge_data'].append(cycle_info)
                    
                    elif 'impedance' in cycle_type.lower():
                        if 'Sense_current' in data.dtype.names:
                            cycle_info['sense_current'] = data['Sense_current'][0].flatten()
                        if 'Battery_current' in data.dtype.names:
                            cycle_info['battery_current'] = data['Battery_current'][0].flatten()
                        if 'Battery_impedance' in data.dtype.names:
                            cycle_info['battery_impedance'] = data['Battery_impedance'][0].flatten()
                        if 'Rectified_impedance' in data.dtype.names:
                            cycle_info['rectified_impedance'] = data['Rectified_impedance'][0].flatten()
                        if 'Re' in data.dtype.names:
                            cycle_info['re'] = float(data['Re'][0][0]) if len(data['Re'][0]) > 0 else np.nan
                        if 'Rct' in data.dtype.names:
                            cycle_info['rct'] = float(data['Rct'][0][0]) if len(data['Rct'][0]) > 0 else np.nan
                        
                        result['impedance_data'].append(cycle_info)
            
    except Exception as e:
        print(f"处理 {battery_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return result


def save_to_excel(battery_data, output_folder):
    """
    将电池数据保存为Excel文件
    
    参数:
        battery_data: 电池数据字典
        output_folder: 输出文件夹路径
    """
    if not battery_data:
        return
    
    battery_name = battery_data['battery_name']
    output_path = os.path.join(output_folder, f"{battery_name}.xlsx")
    
    # 创建Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. 创建总体摘要表
        summary_data = []
        
        # 添加充电循环摘要
        for cycle in battery_data['charge_data']:
            summary_row = {
                '循环编号': cycle['cycle_number'],
                '类型': cycle['type'],
                '环境温度': cycle['ambient_temperature'],
                '数据点数': len(cycle.get('time', []))
            }
            
            if 'voltage' in cycle and len(cycle['voltage']) > 0:
                summary_row['平均电压(V)'] = np.mean(cycle['voltage'])
                summary_row['最大电压(V)'] = np.max(cycle['voltage'])
                summary_row['最小电压(V)'] = np.min(cycle['voltage'])
            
            if 'current' in cycle and len(cycle['current']) > 0:
                summary_row['平均电流(A)'] = np.mean(cycle['current'])
            
            summary_data.append(summary_row)
        
        # 添加放电循环摘要
        for cycle in battery_data['discharge_data']:
            summary_row = {
                '循环编号': cycle['cycle_number'],
                '类型': cycle['type'],
                '环境温度': cycle['ambient_temperature'],
                '数据点数': len(cycle.get('time', []))
            }
            
            if 'voltage' in cycle and len(cycle['voltage']) > 0:
                summary_row['平均电压(V)'] = np.mean(cycle['voltage'])
                summary_row['最大电压(V)'] = np.max(cycle['voltage'])
                summary_row['最小电压(V)'] = np.min(cycle['voltage'])
            
            if 'current' in cycle and len(cycle['current']) > 0:
                summary_row['平均电流(A)'] = np.mean(cycle['current'])
            
            if 'capacity' in cycle and len(cycle['capacity']) > 0:
                summary_row['容量(Ah)'] = cycle['capacity'][-1]
            
            summary_data.append(summary_row)
        
        # 添加阻抗循环摘要
        for cycle in battery_data['impedance_data']:
            summary_row = {
                '循环编号': cycle['cycle_number'],
                '类型': cycle['type'],
                '环境温度': cycle['ambient_temperature'],
                '数据点数': len(cycle.get('battery_impedance', []))
            }
            
            if 're' in cycle:
                summary_row['Re(Ω)'] = cycle['re']
            if 'rct' in cycle:
                summary_row['Rct(Ω)'] = cycle['rct']
            
            summary_data.append(summary_row)
        
        # 保存总体摘要
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='总体摘要', index=False)
        
        # 2. 保存放电容量趋势（重要数据）
        discharge_summary = []
        for cycle in battery_data['discharge_data']:
            if 'capacity' in cycle and len(cycle['capacity']) > 0:
                discharge_summary.append({
                    '循环编号': cycle['cycle_number'],
                    '容量(Ah)': cycle['capacity'][-1],
                    '环境温度(°C)': cycle['ambient_temperature']
                })
        
        if discharge_summary:
            df_discharge = pd.DataFrame(discharge_summary)
            df_discharge.to_excel(writer, sheet_name='放电容量趋势', index=False)
        
        # 3. 保存充电详细数据（选择部分循环）
        charge_samples = [0, len(battery_data['charge_data'])//2, len(battery_data['charge_data'])-1] if battery_data['charge_data'] else []
        for idx in charge_samples:
            if idx < len(battery_data['charge_data']):
                cycle = battery_data['charge_data'][idx]
                if 'time' in cycle and len(cycle['time']) > 0:
                    cycle_df = pd.DataFrame({
                        '时间(s)': cycle['time'],
                        '电压(V)': cycle.get('voltage', []),
                        '电流(A)': cycle.get('current', []),
                        '温度(°C)': cycle.get('temperature', [])
                    })
                    
                    sheet_name = f"充电{cycle['cycle_number']}"[:31]
                    cycle_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 4. 保存放电详细数据（选择部分循环）
        discharge_samples = [0, len(battery_data['discharge_data'])//2, len(battery_data['discharge_data'])-1] if battery_data['discharge_data'] else []
        for idx in discharge_samples:
            if idx < len(battery_data['discharge_data']):
                cycle = battery_data['discharge_data'][idx]
                if 'time' in cycle and len(cycle['time']) > 0:
                    df_dict = {
                        '时间(s)': cycle['time'],
                        '电压(V)': cycle.get('voltage', []),
                        '电流(A)': cycle.get('current', []),
                        '温度(°C)': cycle.get('temperature', [])
                    }
                    
                    if 'capacity' in cycle:
                        df_dict['容量(Ah)'] = cycle['capacity']
                    
                    cycle_df = pd.DataFrame(df_dict)
                    sheet_name = f"放电{cycle['cycle_number']}"[:31]
                    cycle_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 5. 保存阻抗数据摘要
        if battery_data['impedance_data']:
            impedance_summary = []
            for cycle in battery_data['impedance_data']:
                impedance_summary.append({
                    '循环编号': cycle['cycle_number'],
                    'Re(Ω)': cycle.get('re', np.nan),
                    'Rct(Ω)': cycle.get('rct', np.nan),
                    '环境温度(°C)': cycle['ambient_temperature']
                })
            
            if impedance_summary:
                df_impedance = pd.DataFrame(impedance_summary)
                df_impedance.to_excel(writer, sheet_name='阻抗摘要', index=False)
    
    print(f"✓ 已保存: {output_path}")


def convert_all_mat_files(input_folder, output_folder):
    """
    转换文件夹中的所有.mat文件，保持原有目录结构
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有.mat文件
    input_path = Path(input_folder)
    mat_files = list(input_path.rglob('*.mat'))
    
    if not mat_files:
        print(f"在 {input_folder} 中未找到.mat文件")
        return
    
    print(f"找到 {len(mat_files)} 个.mat文件")
    
    # 转换每个文件
    for idx, mat_file in enumerate(mat_files, 1):
        print(f"\n[{idx}/{len(mat_files)}] 处理: {mat_file.name}")
        
        # 计算相对路径，保持目录结构
        try:
            relative_path = mat_file.relative_to(input_path)
            relative_dir = relative_path.parent
            
            # 创建对应的输出目录
            output_dir = Path(output_folder) / relative_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取并保存数据
            battery_data = extract_battery_data(str(mat_file))
            if battery_data:
                save_to_excel(battery_data, str(output_dir))
        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("=" * 60)
    print("MATLAB数据转Excel转换器")
    print("=" * 60)
    
    # 设置输入和输出路径
    # 默认路径 - 可以修改
    input_folder = r"f:\数模\美赛\Data\07281-main\NASA测试数据\NASA_Battery Data Set"
    output_folder = r"f:\数模\美赛\Data\Excel输出"
    
    print(f"\n输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print()
    
    # 执行转换
    convert_all_mat_files(input_folder, output_folder)
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
