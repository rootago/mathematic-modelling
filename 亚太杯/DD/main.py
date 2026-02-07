import pandas as pd
from datetime import datetime
import numpy as np

def process_DD(input_file='input.xlsx', output_file='output.xlsx'):
    try:

        wind_dir_map = {
    "从北方吹来的风": 0,
    "从东北偏北方向吹来的风": 22.5,
    "从东北方吹来的风": 45,
    "从东北偏东方向吹来的风": 67.5,
    "从东方吹来的风": 90,
    "从东南偏东方向吹来的风": 112.5,
    "从东南方吹来的风": 135,
    "从东南偏南方向吹来的风": 157.5,
    "从南方吹来的风": 180,
    "从西南偏南方向吹来的风": 202.5,
    "从西南方吹来的风": 225,
    "从西南偏西方向吹来的风": 247.5,
    "从西方吹来的风": 270,
    "从西北偏西方向吹来的风": 292.5,
    "从西北方吹来的风": 315,
    "从西北偏北方向吹来的风": 337.5
}

        df = pd.read_excel(input_file)

        if 'DD' not in df.columns:
            print("错误：未找到'DD'列")
            return
        
        df['DD'] = df['DD'].replace("无风", np.nan)

        df["DD_deg"] = df['DD'].map(wind_dir_map)

        unmatched = df[df["DD_deg"].isna()]["DD"].unique()
        if len(unmatched) > 0:
            print("⚠️ 以下风向未匹配上，请检查拼写或补充映射表：")
            print(unmatched)
        df = df.drop(columns=["DD"]).rename(columns={"DD_deg": "DD"})
        df.to_excel("output.xlsx", index=False)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    process_DD()