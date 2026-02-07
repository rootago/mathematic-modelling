@echo off
chcp 65001 >nul
echo ====================================
echo 电池数据集批量转换工具
echo ====================================
echo.
echo 本程序将：
echo 1. 自动解压所有zip文件
echo 2. 转换所有.mat文件为Excel格式
echo 3. 保持原有目录结构
echo.
echo 正在启动转换程序...
echo.

python convert_battery_dataset.py

echo.
echo ====================================
echo 按任意键退出...
pause >nul
