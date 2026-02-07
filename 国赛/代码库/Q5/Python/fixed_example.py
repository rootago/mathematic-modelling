"""
修复版本的无人机烟幕遮蔽优化
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FixedSmokeOptimizer:
    """修复版本的烟幕优化器"""
    
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
        
    def get_time_interval(self, t1: float, t2: float, v: float, theta: float,
                         APos0: np.ndarray, BPos0: np.ndarray) -> Tuple[float, float]:
        """计算遮蔽时间区间"""
        
        # 计算导弹方向
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        
        points = self.observation_points
        times = []
        
        for OPos in points:
            t_step = 0.1
            t = 0
            flag1 = 0
            flag2 = 0
            current_t_start = 0
            current_t_end = 0
            
            while t <= 20:
                # 导弹位置
                APos = APos0 + (t + t1 + t2) * ADir * self.AVel
                
                # 烟幕弹位置 - 修复MATLAB中的错误
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
                
                # 遮蔽条件
                if flag1 == 0 and distance1 <= 10:
                    flag1 = 1
                    current_t_start = t + t1 + t2
                    
                if flag1 == 1 and distance1 > 10:
                    current_t_end = t + t1 + t2
                    break
                    
                if distance2 <= 10:
                    flag2 = 1
                    
                if flag2 == 1 and distance2 > 10:
                    current_t_end = t + t1 + t2
                    break
                
                t += t_step
            
            times.append([current_t_start, current_t_end])
        
        times = np.array(times)
        t_start = np.max(times[:, 0])
        t_end = np.min(times[:, 1])
        
        return t_start, t_end
    
    def merge_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """合并重叠的时间区间"""
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
        
        return np.array(merged)
    
    def objective_function(self, x: np.ndarray, drone_idx: int, 
                         existing_intervals: List[np.ndarray]) -> float:
        """目标函数"""
        penalty = 0
        current_intervals = []
        
        # 计算当前无人机对三枚导弹的遮蔽区间
        for i in range(3):
            t1 = x[i*2]
            t2 = x[i*2 + 1]
            v = x[6]
            theta = x[7]
            
            t_start, t_end = self.get_time_interval(t1, t2, v, theta,
                                                  self.APos0[i], 
                                                  self.BPos0[drone_idx])
            
            if t_start < t_end:
                current_intervals.append([t_start, t_end])
            else:
                current_intervals.append([])
        
        # 约束条件检查 - 修复约束逻辑
        for i in range(2):
            t_prev = x[i*2]
            t_post = x[(i+1)*2]
            if abs(t_post - t_prev) < 1:
                penalty += 10000
        
        # 计算总遮蔽时间
        total_time = 0
        
        for i in range(3):
            if len(current_intervals[i]) > 0:
                current_interval = np.array([current_intervals[i]])
                
                if len(existing_intervals[i]) == 0:
                    merged = self.merge_intervals(current_interval)
                else:
                    combined = np.vstack([existing_intervals[i], current_interval])
                    merged = self.merge_intervals(combined)
                
                for interval in merged:
                    total_time += (interval[1] - interval[0])
            else:
                for interval in existing_intervals[i]:
                    total_time += (interval[1] - interval[0])
        
        return -(total_time - penalty)
    
    def optimize_single_drone(self, drone_idx: int, 
                            existing_intervals: List[np.ndarray]) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """优化单架无人机"""
        
        bounds = [(0, 7), (1.4, 7), (0, 7), (1.4, 7), (0, 7), (1.4, 7), (70, 140), (0, np.pi)]
        
        def obj_func(x):
            return self.objective_function(x, drone_idx, existing_intervals)
        
        # 使用多种优化方法尝试
        methods = [
            ('differential_evolution', lambda: differential_evolution(
                obj_func, bounds, maxiter=2000, popsize=15, seed=42+drone_idx,
                atol=1e-6, tol=1e-6
            )),
            ('minimize', lambda: minimize(
                obj_func, x0=np.array([1.5, 3.6, 1.5, 3.6, 1.5, 3.6, 120, np.pi/4]),
                bounds=bounds, method='L-BFGS-B'
            ))
        ]
        
        best_result = None
        best_fval = np.inf
        
        for method_name, method_func in methods:
            try:
                result = method_func()
                if result.fun < best_fval:
                    best_fval = result.fun
                    best_result = result
            except Exception as e:
                print(f"方法 {method_name} 失败: {e}")
                continue
        
        if best_result is None:
            # 如果所有方法都失败，使用随机解
            x_opt = np.array([1.5, 3.6, 1.5, 3.6, 1.5, 3.6, 120, np.pi/4])
            fval = obj_func(x_opt)
        else:
            x_opt = best_result.x
            fval = best_result.fun
        
        # 计算新的区间
        new_intervals = []
        
        for i in range(3):
            t1 = x_opt[i*2]
            t2 = x_opt[i*2 + 1]
            v = x_opt[6]
            theta = x_opt[7]
            
            t_start, t_end = self.get_time_interval(t1, t2, v, theta,
                                                  self.APos0[i], 
                                                  self.BPos0[drone_idx])
            
            if t_start < t_end:
                new_intervals.append(np.array([[t_start, t_end]]))
            else:
                new_intervals.append(np.array([]))
        
        # 更新区间
        updated_intervals = []
        for i in range(3):
            if len(new_intervals[i]) > 0:
                if len(existing_intervals[i]) == 0:
                    updated_intervals.append(self.merge_intervals(new_intervals[i]))
                else:
                    combined = np.vstack([existing_intervals[i], new_intervals[i]])
                    updated_intervals.append(self.merge_intervals(combined))
            else:
                updated_intervals.append(existing_intervals[i])
        
        return x_opt, fval, updated_intervals
    
    def test_single_calculation(self):
        """测试单次计算"""
        print("=== 测试单次计算 ===")
        
        # 使用不同的测试参数
        test_params = [
            (1.5, 3.6, 120, np.pi/4),
            (2.0, 4.0, 100, np.pi/3),
            (1.0, 2.0, 140, np.pi/6)
        ]
        
        for i, (t1, t2, v, theta) in enumerate(test_params):
            print(f"\n测试参数组 {i+1}: t1={t1}, t2={t2}, v={v}, theta={theta}")
            
            for j in range(3):
                APos0 = self.APos0[j]
                BPos0 = self.BPos0[0]
                
                t_start, t_end = self.get_time_interval(t1, t2, v, theta, APos0, BPos0)
                
                print(f"  导弹{j+1}: [{t_start:.2f}, {t_end:.2f}]", end="")
                if t_start < t_end:
                    print(f" (有效, 时长: {t_end - t_start:.2f})")
                else:
                    print(" (无效)")
    
    def optimize_all_drones(self):
        """优化所有无人机"""
        print("开始优化所有无人机...")
        
        # 初始化区间
        intervals = [np.array([]) for _ in range(3)]
        solutions = []
        
        for drone_idx in range(5):
            print(f"\n优化第 {drone_idx + 1} 架无人机")
            
            # 优化当前无人机
            x_opt, fval, intervals = self.optimize_single_drone(drone_idx, intervals)
            
            # 保存解
            solutions.append(x_opt)
            
            # 输出结果
            print(f"第{drone_idx + 1}架无人机最优解：")
            print(f"  t1, t2, t1, t2, t1, t2, v, theta = {x_opt}")
            print(f"第{drone_idx + 1}架无人机目标值：{-fval:.4f}")
        
        # 输出最终结果
        print("\n最终结果")
        for i in range(3):
            print(f"导弹{i+1}的累积遮蔽时间区间：")
            if len(intervals[i]) > 0:
                total_time = 0
                for interval in intervals[i]:
                    print(f"[{interval[0]:.2f}, {interval[1]:.2f}] ", end="")
                    total_time += (interval[1] - interval[0])
                print(f"\n导弹{i+1}总遮蔽时间：{total_time:.4f}")
            else:
                print("无遮蔽区间")
        
        return solutions, intervals


def main():
    """主函数"""
    optimizer = FixedSmokeOptimizer()
    
    # 先测试单次计算
    optimizer.test_single_calculation()
    
    # 然后运行优化
    solutions, intervals = optimizer.optimize_all_drones()
    
    # 简单的结果可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        ax = axes[i]
        if len(intervals[i]) > 0:
            for j, interval in enumerate(intervals[i]):
                ax.barh(j, interval[1] - interval[0], left=interval[0], 
                       height=0.8, alpha=0.7)
            ax.set_xlabel('时间')
            ax.set_ylabel('区间')
            ax.set_title(f'导弹{i+1}遮蔽时间区间')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无遮蔽区间', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'导弹{i+1}遮蔽时间区间')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
