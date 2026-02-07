# 无人机烟幕遮蔽优化系统

这是将MATLAB代码转换为Python的版本，使用scipy库进行优化，避免了MATLAB全局变量的缓存问题。

## 文件说明

- `optimization_system.py`: 完整的优化系统类，包含所有功能
- `simple_example.py`: 简化版本，更容易理解和运行
- `requirements.txt`: 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行简化版本（推荐）

```python
python simple_example.py
```

### 运行完整版本

```python
python optimization_system.py
```

## 主要改进

1. **消除全局变量**: 使用类属性替代MATLAB的全局变量
2. **使用scipy优化**: 用`differential_evolution`替代MATLAB的`simulannealbnd`
3. **更好的数据结构**: 使用numpy数组和列表替代MATLAB的cell数组
4. **模块化设计**: 将功能分解为独立的方法

## 核心功能

1. **generate_points()**: 生成观测点坐标
2. **get_time_interval()**: 计算烟幕遮蔽时间区间
3. **merge_intervals()**: 合并重叠的时间区间
4. **objective_function()**: 目标函数（最大化总遮蔽时间）
5. **optimize_single_drone()**: 优化单架无人机
6. **optimize_all_drones()**: 优化所有无人机

## 优化参数

- 变量: [t1, t2, t1, t2, t1, t2, v, theta]
- 边界: t1,t2∈[0,7], t2∈[1.4,7], v∈[70,140], theta∈[0,π]
- 算法: 差分进化算法（differential_evolution）

## 输出结果

程序会输出每架无人机的最优解和最终的三枚导弹累积遮蔽时间区间。
