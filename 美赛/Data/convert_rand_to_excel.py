"""
仅转换 Matlab/11.+Randomized+Battery+Usage+Data+Set 数据集
先解压所有 zip，再按原目录结构将 .mat 转为 Excel
"""

from pathlib import Path

from convert_battery_dataset import extract_zip_files, convert_all_mat_files


def main():
    base = Path(__file__).resolve().parent
    input_dir = base / "Matlab" / "11.+Randomized+Battery+Usage+Data+Set"
    output_dir = base / "Excel输出" / "数据集3-RandomizedBattery"

    print("=" * 60)
    print("Randomized Battery Usage 数据集 → Excel")
    print("=" * 60)
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print()

    if not input_dir.exists():
        print(f"目录不存在: {input_dir}")
        return

    extract_zip_files(str(input_dir))
    convert_all_mat_files(str(input_dir), str(output_dir))

    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
