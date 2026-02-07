"""
电池数据集批量转换工具
自动解压zip文件并转换所有.mat文件为Excel格式
保持原有目录结构
"""

import scipy.io as sio
import pandas as pd
import numpy as np
import os
import zipfile
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
    
    # 创建Excel writer（必须至少有一个可见工作表，否则 openpyxl 报错）
    sheet_written = False
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
            sheet_written = True
        
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
            sheet_written = True
        
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
                    sheet_written = True
        
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
                    sheet_written = True
        
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
                sheet_written = True
        
        # 若未写入任何表（如 Randomized 等非 NASA 结构），至少写一页说明，避免 openpyxl 报错
        if not sheet_written:
            pd.DataFrame([{'说明': '未提取到符合NASA结构的数据，该.mat格式可能不同'}]).to_excel(
                writer, sheet_name='说明', index=False
            )
        # 确保至少有一个可见且激活的工作表（避免 openpyxl "At least one sheet must be visible"）
        wb = writer.book
        if wb.worksheets:
            first = wb.worksheets[0]
            first.sheet_state = 'visible'
            wb.active = 0
    
    print(f"✓ 已保存: {output_path}")


def extract_zip_files(input_folder, extract_to=None):
    """
    解压文件夹中的所有zip文件
    
    参数:
        input_folder: 包含zip文件的文件夹
        extract_to: 解压目标文件夹（如果为None，则解压到zip文件所在位置）
    """
    input_path = Path(input_folder)
    zip_files = list(input_path.glob('*.zip'))
    
    if not zip_files:
        print(f"在 {input_folder} 中未找到zip文件")
        return
    
    print(f"\n找到 {len(zip_files)} 个压缩文件，开始解压...")
    
    for idx, zip_file in enumerate(zip_files, 1):
        # 确定解压路径
        if extract_to:
            extract_path = Path(extract_to) / zip_file.stem
        else:
            extract_path = zip_file.parent / zip_file.stem
        
        # 如果已经解压过，跳过
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"[{idx}/{len(zip_files)}] 跳过（已存在）: {zip_file.name}")
            continue
        
        print(f"[{idx}/{len(zip_files)}] 解压中: {zip_file.name}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"  ✓ 解压完成: {extract_path}")
        except Exception as e:
            print(f"  ✗ 解压失败: {str(e)}")


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
    
    print(f"\n找到 {len(mat_files)} 个.mat文件，开始转换...")
    
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
    print("=" * 70)
    print("电池数据集批量转换工具")
    print("=" * 70)
    
    # 设置路径
    # 数据集1: Matlab/07281-main/NASA测试数据
    dataset1_input = r"f:\数模\美赛\Data\Matlab\07281-main\NASA测试数据\NASA_Battery Data Set"
    dataset1_output = r"f:\数模\美赛\Data\Excel输出\数据集1-NASA测试数据"
    
    # 数据集2: Matlab/5.+Battery+Data+Set
    dataset2_input = r"f:\数模\美赛\Data\Matlab\5.+Battery+Data+Set\5. Battery Data Set"
    dataset2_output = r"f:\数模\美赛\Data\Excel输出\数据集2-BatteryAgingARC"
    
    print("\n【数据集1】NASA测试数据")
    print(f"输入: {dataset1_input}")
    print(f"输出: {dataset1_output}")
    convert_all_mat_files(dataset1_input, dataset1_output)
    
    print("\n" + "=" * 70)
    print("【数据集2】BatteryAgingARC数据集")
    print(f"输入: {dataset2_input}")
    print(f"输出: {dataset2_output}")
    
    # 先解压zip文件
    extract_zip_files(dataset2_input)
    
    # 然后转换所有.mat文件
    convert_all_mat_files(dataset2_input, dataset2_output)
    
    # 数据集3: Matlab/11.+Randomized+Battery+Usage+Data+Set
    dataset3_input = Path(__file__).resolve().parent / "Matlab" / "11.+Randomized+Battery+Usage+Data+Set"
    dataset3_output = Path(__file__).resolve().parent / "Excel输出" / "数据集3-RandomizedBattery"
    print("\n" + "=" * 70)
    print("【数据集3】Randomized Battery Usage 数据集")
    print(f"输入: {dataset3_input}")
    print(f"输出: {dataset3_output}")
    if dataset3_input.exists():
        extract_zip_files(str(dataset3_input))
        convert_all_mat_files(str(dataset3_input), str(dataset3_output))
    else:
        print(f"跳过（目录不存在）: {dataset3_input}")
    
    # 数据集4: Matlab/archive (MATR batchdata)
    try:
        import sys
        archive_script = Path(__file__).resolve().parent / "convert_archive_to_excel.py"
        if archive_script.exists():
            print("\n" + "=" * 70)
            print("【数据集4】Matlab/archive (MATR batchdata)")
            archive_input = Path(__file__).resolve().parent / "Matlab" / "archive"
            archive_output = Path(__file__).resolve().parent / "Excel输出" / "archive"
            if archive_input.exists():
                print(f"输入: {archive_input}")
                print(f"输出: {archive_output}")
                from convert_archive_to_excel import convert_archive_folder
                convert_archive_folder(archive_input, archive_output)
            else:
                print(f"跳过（目录不存在）: {archive_input}")
    except Exception as e:
        print(f"\narchive 转换跳过或失败: {e}")
    
    print("\n" + "=" * 70)
    print("✓ 全部转换完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
