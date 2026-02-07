"""
无人机烟幕遮蔽优化系统
使用scipy库替代MATLAB的全局变量和优化算法
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SmokeOptimizationSystem:
    """无人机烟幕遮蔽优化系统"""
    
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
        
        # 导弹参数
        self.AVel = 300  # 导弹速度
        self.g = 9.8     # 重力加速度
        
        # 全局遮蔽区间存储（替代MATLAB全局变量）
        self.global_smoke_intervals = [[] for _ in range(3)]  # 3枚导弹的遮蔽区间
        
        # 优化参数
        self.num_drones = 5
        self.optimal_solutions = []
        
    def generate_points(self) -> np.ndarray:
        """生成观测点坐标"""
        points = []
        # O1
        points.append([0, 193, 10])
        # 其他点被注释掉了，只保留O1
        return np.array(points)
    
    def get_time(self, t1: float, t2: float, v: float, theta: float, 
                 APos0: np.ndarray, BPos0: np.ndarray) -> Tuple[float, float]:
        """
        计算烟幕遮蔽时间区间
        
        Args:
            t1: 发射延迟时间
            t2: 飞行时间
            v: 发射速度
            theta: 发射角度
            APos0: 导弹初始位置
            BPos0: 无人机初始位置
            
        Returns:
            (t_start, t_end): 遮蔽时间区间
        """
        # 计算导弹方向
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        
        # 获取观测点
        points = self.generate_points()
        times = []
        
        for i in range(len(points)):
            OPos = points[i]
            
            # 仿真参数
            t_step = 0.1
            t = 0
            flag1 = 0
            flag2 = 0
            current_t_start = 0
            current_t_end = 0
            
            while t <= 20:
                # 计算导弹位置
                APos = APos0 + (t + t1 + t2) * ADir * self.AVel
                
                # 计算烟幕弹位置
                DPos = np.array([
                    BPos0[0] + v * (t1 + t2) * np.cos(theta),
                    BPos0[1] + v * (t1 + t2) * np.sin(theta),
                    BPos0[2] - 0.5 * self.g * t2**2 - 3 * t
                ])
                
                # 计算距离
                OA = APos - OPos
                OD = DPos - OPos
                
                # 计算点到直线的距离
                if np.linalg.norm(OA) > 1e-10:
                    distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
                else:
                    distance1 = np.inf
                    
                distance2 = np.linalg.norm(APos - DPos)
                
                # 检查遮蔽条件
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
    
    def merge_time(self, times: np.ndarray) -> np.ndarray:
        """
        合并重叠的时间区间
        
        Args:
            times: 时间区间数组，形状为(n, 2)
            
        Returns:
            合并后的时间区间数组
        """
        if len(times) == 0:
            return np.array([])
            
        # 过滤掉无效区间
        times = times[times[:, 0] > 0]
        
        if len(times) == 0:
            return np.array([])
        
        # 按开始时间排序
        times = times[np.argsort(times[:, 0])]
        
        # 合并区间
        merged = [times[0]]
        
        for i in range(1, len(times)):
            curr = times[i]
            last = merged[-1]
            
            if curr[0] <= last[1]:
                # 区间重叠，合并
                merged[-1][1] = max(curr[1], last[1])
            else:
                # 区间不重叠，添加新区间
                merged.append(curr)
        
        return np.array(merged)
    
    def update_global_intervals(self, x: np.ndarray, curr_drone: int):
        """
        更新全局遮蔽区间
        
        Args:
            x: 优化变量 [t1, t2, t1, t2, t1, t2, v, theta]
            curr_drone: 当前无人机索引
        """
        current_intervals = [[] for _ in range(3)]
        
        # 计算当前无人机对三枚导弹的遮蔽区间
        for i in range(0, 5, 2):  # 0, 2, 4
            t1 = x[i]
            t2 = x[i + 1]
            v = x[6]
            theta = x[7]
            missile_idx = i // 2  # 0, 1, 2
            
            t_start, t_end = self.get_time(t1, t2, v, theta, 
                                         self.APos0[missile_idx], 
                                         self.BPos0[curr_drone])
            
            if t_start < t_end:
                current_intervals[missile_idx].append([t_start, t_end])
        
        # 更新全局区间
        for missile_idx in range(3):
            if len(current_intervals[missile_idx]) > 0:
                current_intervals[missile_idx] = np.array(current_intervals[missile_idx])
                
                if len(self.global_smoke_intervals[missile_idx]) == 0:
                    # 如果全局区间为空，直接使用当前区间
                    self.global_smoke_intervals[missile_idx] = self.merge_time(current_intervals[missile_idx])
                else:
                    # 合并全局区间和当前区间
                    combined = np.vstack([self.global_smoke_intervals[missile_idx], 
                                        current_intervals[missile_idx]])
                    self.global_smoke_intervals[missile_idx] = self.merge_time(combined)
    
    def objective_function(self, x: np.ndarray, curr_drone: int) -> float:
        """
        目标函数：计算总遮蔽时间
        
        Args:
            x: 优化变量 [t1, t2, t1, t2, t1, t2, v, theta]
            curr_drone: 当前无人机索引
            
        Returns:
            目标函数值（负值，因为要最大化）
        """
        penalty = 0
        current_intervals = [[] for _ in range(3)]
        
        # 计算当前无人机对三枚导弹的遮蔽区间
        for i in range(0, 5, 2):  # 0, 2, 4
            t1 = x[i]
            t2 = x[i + 1]
            v = x[6]
            theta = x[7]
            missile_idx = i // 2
            
            t_start, t_end = self.get_time(t1, t2, v, theta, 
                                         self.APos0[missile_idx], 
                                         self.BPos0[curr_drone])
            
            if t_start < t_end:
                current_intervals[missile_idx].append([t_start, t_end])
        
        # 约束条件检查
        for i in range(0, 3, 2):  # 0, 2
            t_prev = x[i]
            t_post = x[i + 2]
            if abs(t_post - t_prev) < 1:
                penalty += 10000
        
        # 计算总遮蔽时间
        total_time = 0
        
        for missile_idx in range(3):
            if len(current_intervals[missile_idx]) > 0:
                current_intervals[missile_idx] = np.array(current_intervals[missile_idx])
                
                if len(self.global_smoke_intervals[missile_idx]) == 0:
                    # 如果全局区间为空，只计算当前区间的时间
                    merged_intervals = self.merge_time(current_intervals[missile_idx])
                else:
                    # 合并全局区间和当前区间，但不保存到全局变量
                    combined = np.vstack([self.global_smoke_intervals[missile_idx], 
                                        current_intervals[missile_idx]])
                    merged_intervals = self.merge_time(combined)
                
                # 计算合并后区间的总时间
                for interval in merged_intervals:
                    total_time += (interval[1] - interval[0])
            else:
                # 如果当前无人机对该导弹没有贡献，只计算全局区间的时间
                for interval in self.global_smoke_intervals[missile_idx]:
                    total_time += (interval[1] - interval[0])
        
        return -(total_time - penalty)
    
    def optimize_single_drone(self, drone_idx: int) -> Tuple[np.ndarray, float]:
        """
        优化单架无人机
        
        Args:
            drone_idx: 无人机索引
            
        Returns:
            (最优解, 最优值)
        """
        # 变量边界
        bounds = [(0, 7), (1.4, 7), (0, 7), (1.4, 7), (0, 7), (1.4, 7), (70, 140), (0, np.pi)]
        
        # 使用差分进化算法（类似模拟退火）
        result = differential_evolution(
            lambda x: self.objective_function(x, drone_idx),
            bounds,
            maxiter=40000,
            popsize=15,
            seed=42,
            atol=1e-12,
            tol=1e-12
        )
        
        return result.x, result.fun
    
    def optimize_all_drones(self):
        """优化所有无人机"""
        print("开始优化所有无人机...")
        
        for drone_idx in range(self.num_drones):
            print(f"\n优化第 {drone_idx + 1} 架无人机")
            
            # 优化当前无人机
            x_opt, fval = self.optimize_single_drone(drone_idx)
            
            # 更新全局区间
            self.update_global_intervals(x_opt, drone_idx)
            
            # 保存解
            self.optimal_solutions.append(x_opt)
            
            # 输出结果
            print(f"第{drone_idx + 1}架无人机最优解：")
            print(x_opt)
            print(f"第{drone_idx + 1}架无人机目标值：{-fval:.4f}")
        
        # 输出最终结果
        print("\n最终结果")
        for i in range(3):
            print(f"导弹{i+1}的累积遮蔽时间区间：")
            if len(self.global_smoke_intervals[i]) > 0:
                total_time = 0
                for interval in self.global_smoke_intervals[i]:
                    print(f"[{interval[0]:.2f}, {interval[1]:.2f}] ", end="")
                    total_time += (interval[1] - interval[0])
                print(f"\n导弹{i+1}总遮蔽时间：{total_time:.4f}")
            else:
                print("无遮蔽区间")
    
    def visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i in range(3):
            ax = axes[i]
            if len(self.global_smoke_intervals[i]) > 0:
                for j, interval in enumerate(self.global_smoke_intervals[i]):
                    ax.barh(j, interval[1] - interval[0], left=interval[0], 
                           height=0.8, alpha=0.7, label=f'区间{j+1}')
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


def main():
    """主函数"""
    # 创建优化系统
    system = SmokeOptimizationSystem()
    
    # 执行优化
    system.optimize_all_drones()
    
    # 可视化结果
    system.visualize_results()


if __name__ == "__main__":
    main()
