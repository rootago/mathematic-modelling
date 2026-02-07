import pandas as pd
from datetime import datetime
import numpy as np

def process_excel_data(input_file='input.xlsx', output_file='output1.xlsx'):
    """
    从Excel文件读取数据，按日期合并求均值并导出
    
    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)
        
        # 确保Time列存在
        if 'Time' not in df.columns:
            print("错误: 未找到'Time'列")
            return
        
        # 转换Time列为datetime格式
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M')
        
        # 提取日期（去掉时间部分）
        df['Date'] = df['Time'].dt.date
        
        # 定义需要计算均值的数值列
        columns = ['T', 'Po', 'P', 'Pa', 'U', 'Ff', 'RRR','DD']

        
        # 按日期分组并计算均值
        daily_avg = df.groupby('Date')[columns].mean().reset_index()
        
        # 添加日期格式化列用于显示
        daily_avg['Date_Formatted'] = daily_avg['Date'].astype(str)
        
        # 重新排列列的顺序，将格式化的日期放在前面
        columns_order = ['Date_Formatted'] + columns
        daily_avg = daily_avg[columns_order]
        
        daily_avg.rename(columns={'Date_Formatted': 'Date'}, inplace=True)
        
        
        # 保存到Excel文件
        daily_avg.to_excel(output_file, index=False)
        
        print(f"数据已成功保存到 {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    process_excel_data()
    