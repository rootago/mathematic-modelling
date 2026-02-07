"""
调试版本 - 检查转换问题
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DebugSmokeOptimizer:
    """调试版本的烟幕优化器"""
    
    def __init__(self):
        # 导弹初始位置
        self.APos0 = np.array([
            [20000, 0, 2000],
            [19000, 600, 2100], 
            [18000, -600, 1900]
        ])
        
        # 无人机初始位置
        self.BPos0 = np.array([
            [17800, 0, 1800],
            [12000, 1400, 1400],
            [6000, -3000, 700],
            [11000, 2000, 1800],
            [13000, -2000, 1300]
        ])
        
        # 物理参数
        self.AVel = 300  # 导弹速度
        self.g = 9.8     # 重力加速度
        
        # 观测点
        self.observation_points = np.array([[0, 193, 10]])
        
    def get_time_interval_debug(self, t1: float, t2: float, v: float, theta: float,
                               APos0: np.ndarray, BPos0: np.ndarray) -> Tuple[float, float]:
        """调试版本的时间区间计算"""
        
        print(f"  调试: t1={t1}, t2={t2}, v={v}, theta={theta}")
        print(f"  调试: APos0={APos0}, BPos0={BPos0}")
        
        # 计算导弹方向
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        print(f"  调试: alpha1={alpha1}, ADir={ADir}")
        
        points = self.observation_points
        times = []
        
        for i, OPos in enumerate(points):
            print(f"  调试: 观测点{i+1} = {OPos}")
            
            t_step = 0.1
            t = 0
            flag1 = 0
            flag2 = 0
            current_t_start = 0
            current_t_end = 0
            
            step_count = 0
            while t <= 20 and step_count < 10:  # 限制步数用于调试
                # 导弹位置
                APos = APos0 + (t + t1 + t2) * ADir * self.AVel
                
                # 烟幕弹位置
                DPos = np.array([
                    BPos0[0] + v * (t1 + t2) * np.cos(theta),
                    BPos0[1] + v * (t1 + t2) * np.sin(theta),
                    BPos0[2] - 0.5 * self.g * t2**2 - 3 * t
                ])
                
                # 计算距离
                OA = APos - OPos
                OD = DPos - OPos
                
                if np.linalg.norm(OA) > 1e-10:
                    distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
                else:
                    distance1 = np.inf
                    
                distance2 = np.linalg.norm(APos - DPos)
                
                if step_count < 5:  # 只打印前几步
                    print(f"    步骤{step_count}: t={t:.2f}, APos={APos}, DPos={DPos}")
                    print(f"    distance1={distance1:.2f}, distance2={distance2:.2f}")
                
                # 遮蔽条件
                if flag1 == 0 and distance1 <= 10:
                    flag1 = 1
                    current_t_start = t + t1 + t2
                    print(f"    开始遮蔽: t_start={current_t_start:.2f}")
                    
                if flag1 == 1 and distance1 > 10:
                    current_t_end = t + t1 + t2
                    print(f"    结束遮蔽: t_end={current_t_end:.2f}")
                    break
                    
                if distance2 <= 10:
                    flag2 = 1
                    print(f"    距离2遮蔽开始")
                    
                if flag2 == 1 and distance2 > 10:
                    current_t_end = t + t1 + t2
                    print(f"    距离2遮蔽结束: t_end={current_t_end:.2f}")
                    break
                
                t += t_step
                step_count += 1
            
            print(f"  调试: 最终时间区间 = [{current_t_start:.2f}, {current_t_end:.2f}]")
            times.append([current_t_start, current_t_end])
        
        times = np.array(times)
        t_start = np.max(times[:, 0])
        t_end = np.min(times[:, 1])
        
        print(f"  调试: 合并后时间区间 = [{t_start:.2f}, {t_end:.2f}]")
        
        return t_start, t_end
    
    def test_single_calculation(self):
        """测试单次计算"""
        print("=== 测试单次计算 ===")
        
        # 使用MATLAB中的测试参数
        t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
        APos0 = self.APos0[0]
        BPos0 = self.BPos0[0]
        
        print(f"测试参数: t1={t1}, t2={t2}, v={v}, theta={theta}")
        print(f"导弹位置: {APos0}")
        print(f"无人机位置: {BPos0}")
        
        t_start, t_end = self.get_time_interval_debug(t1, t2, v, theta, APos0, BPos0)
        
        print(f"结果: t_start={t_start:.2f}, t_end={t_end:.2f}")
        
        if t_start >= t_end:
            print("警告: 时间区间无效!")
        else:
            print(f"遮蔽时间: {t_end - t_start:.2f}")
    
    def test_optimization_debug(self):
        """调试优化过程"""
        print("\n=== 调试优化过程 ===")
        
        # 测试目标函数
        x_test = np.array([1.5, 3.6, 1.5, 3.6, 1.5, 3.6, 120, np.pi/4])
        existing_intervals = [np.array([]) for _ in range(3)]
        
        print("测试目标函数...")
        fval = self.objective_function_debug(x_test, 0, existing_intervals)
        print(f"目标函数值: {fval}")
    
    def objective_function_debug(self, x: np.ndarray, drone_idx: int, 
                               existing_intervals: List[np.ndarray]) -> float:
        """调试版本的目标函数"""
        print(f"\n--- 目标函数调试 (无人机{drone_idx+1}) ---")
        print(f"输入参数: {x}")
        
        penalty = 0
        current_intervals = []
        
        # 计算当前无人机对三枚导弹的遮蔽区间
        for i in range(3):
            t1 = x[i*2]
            t2 = x[i*2 + 1]
            v = x[6]
            theta = x[7]
            
            print(f"\n导弹{i+1}:")
            t_start, t_end = self.get_time_interval_debug(t1, t2, v, theta,
                                                        self.APos0[i], 
                                                        self.BPos0[drone_idx])
            
            if t_start < t_end:
                current_intervals.append([t_start, t_end])
                print(f"  有效区间: [{t_start:.2f}, {t_end:.2f}]")
            else:
                current_intervals.append([])
                print(f"  无效区间")
        
        # 约束条件检查
        for i in range(2):
            t_prev = x[i*2]
            t_post = x[(i+1)*2]
            if abs(t_post - t_prev) < 1:
                penalty += 10000
                print(f"  约束违反: |{t_post:.2f} - {t_prev:.2f}| < 1")
        
        # 计算总遮蔽时间
        total_time = 0
        
        for i in range(3):
            if len(current_intervals[i]) > 0:
                current_interval = np.array([current_intervals[i]])
                
                if len(existing_intervals[i]) == 0:
                    merged = self.merge_intervals_debug(current_interval)
                else:
                    combined = np.vstack([existing_intervals[i], current_interval])
                    merged = self.merge_intervals_debug(combined)
                
                for interval in merged:
                    total_time += (interval[1] - interval[0])
            else:
                for interval in existing_intervals[i]:
                    total_time += (interval[1] - interval[0])
        
        print(f"总遮蔽时间: {total_time:.4f}")
        print(f"惩罚: {penalty:.4f}")
        print(f"最终目标值: {-(total_time - penalty):.4f}")
        
        return -(total_time - penalty)
    
    def merge_intervals_debug(self, intervals: np.ndarray) -> np.ndarray:
        """调试版本的区间合并"""
        print(f"  合并区间: {intervals}")
        
        if len(intervals) == 0:
            return np.array([])
            
        # 过滤无效区间
        intervals = intervals[intervals[:, 0] > 0]
        
        if len(intervals) == 0:
            return np.array([])
        
        # 排序
        intervals = intervals[np.argsort(intervals[:, 0])]
        
        # 合并
        merged = [intervals[0]]
        for i in range(1, len(intervals)):
            curr = intervals[i]
            last = merged[-1]
            
            if curr[0] <= last[1]:
                merged[-1][1] = max(curr[1], last[1])
            else:
                merged.append(curr)
        
        result = np.array(merged)
        print(f"  合并结果: {result}")
        return result


def main():
    """主调试函数"""
    optimizer = DebugSmokeOptimizer()
    
    # 测试单次计算
    optimizer.test_single_calculation()
    
    # 测试目标函数
    optimizer.test_optimization_debug()


if __name__ == "__main__":
    main()
