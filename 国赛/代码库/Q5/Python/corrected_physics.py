"""
修正物理模型的烟幕优化器
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CorrectedPhysicsOptimizer:
    """修正物理模型的烟幕优化器"""
    
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
        
    def get_time_interval_corrected(self, t1: float, t2: float, v: float, theta: float,
                                   APos0: np.ndarray, BPos0: np.ndarray) -> Tuple[float, float]:
        """修正版本的时间区间计算"""
        
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
                # 导弹位置 - 修正理解
                # 导弹在t1时刻开始飞行，t2是导弹的飞行时间
                APos = APos0 + (t + t1 + t2) * ADir * self.AVel
                
                # 烟幕弹位置 - 修正理解
                # 烟幕弹在t1时刻发射，t2是烟幕弹的飞行时间
                # 发射后，烟幕弹在t时间内下降
                DPos = np.array([
                    BPos0[0] + v * (t1 + t2) * np.cos(theta),
                    BPos0[1] + v * (t1 + t2) * np.sin(theta),
                    BPos0[2] - 0.5 * self.g * t2**2 - 3 * t  # 3*t是下降速度
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
    
    def test_physics_understanding(self):
        """测试物理理解"""
        print("=== 测试物理理解 ===")
        
        # 使用MATLAB中的参数
        t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
        APos0 = self.APos0[0]
        BPos0 = self.BPos0[0]
        
        print(f"参数: t1={t1}, t2={t2}, v={v}, theta={theta}")
        print(f"导弹初始位置: {APos0}")
        print(f"无人机初始位置: {BPos0}")
        print(f"观测点: {self.observation_points[0]}")
        
        # 分析几个关键时间点
        key_times = [0, 1, 2, 5, 10]
        
        print(f"\n关键时间点分析:")
        print(f"{'时间':<6} {'导弹位置':<30} {'烟幕弹位置':<30} {'距离1':<8} {'距离2':<8}")
        print("-" * 90)
        
        for t in key_times:
            # 导弹位置
            alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
            ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
            APos = APos0 + (t + t1 + t2) * ADir * self.AVel
            
            # 烟幕弹位置
            DPos = np.array([
                BPos0[0] + v * (t1 + t2) * np.cos(theta),
                BPos0[1] + v * (t1 + t2) * np.sin(theta),
                BPos0[2] - 0.5 * self.g * t2**2 - 3 * t
            ])
            
            # 计算距离
            OPos = self.observation_points[0]
            OA = APos - OPos
            OD = DPos - OPos
            
            if np.linalg.norm(OA) > 1e-10:
                distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
            else:
                distance1 = np.inf
                
            distance2 = np.linalg.norm(APos - DPos)
            
            print(f"{t:<6.1f} {str(APos):<30} {str(DPos):<30} {distance1:<8.1f} {distance2:<8.1f}")
        
        # 检查是否有满足条件的时刻
        print(f"\n详细搜索满足条件的时刻:")
        found = False
        for t in np.arange(0, 20.1, 0.01):
            APos = APos0 + (t + t1 + t2) * ADir * self.AVel
            DPos = np.array([
                BPos0[0] + v * (t1 + t2) * np.cos(theta),
                BPos0[1] + v * (t1 + t2) * np.sin(theta),
                BPos0[2] - 0.5 * self.g * t2**2 - 3 * t
            ])
            
            OPos = self.observation_points[0]
            OA = APos - OPos
            OD = DPos - OPos
            
            if np.linalg.norm(OA) > 1e-10:
                distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
            else:
                distance1 = np.inf
                
            distance2 = np.linalg.norm(APos - DPos)
            
            if distance1 <= 10 or distance2 <= 10:
                print(f"t={t:.2f}: 距离1={distance1:.2f}, 距离2={distance2:.2f} - 满足条件!")
                found = True
                break
        
        if not found:
            print("在0-20秒内没有找到满足条件的时刻")
            
            # 寻找最小距离
            min_dist1 = np.inf
            min_dist2 = np.inf
            min_t = 0
            
            for t in np.arange(0, 20.1, 0.01):
                APos = APos0 + (t + t1 + t2) * ADir * self.AVel
                DPos = np.array([
                    BPos0[0] + v * (t1 + t2) * np.cos(theta),
                    BPos0[1] + v * (t1 + t2) * np.sin(theta),
                    BPos0[2] - 0.5 * self.g * t2**2 - 3 * t
                ])
                
                OPos = self.observation_points[0]
                OA = APos - OPos
                OD = DPos - OPos
                
                if np.linalg.norm(OA) > 1e-10:
                    distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
                else:
                    distance1 = np.inf
                    
                distance2 = np.linalg.norm(APos - DPos)
                
                if distance1 < min_dist1:
                    min_dist1 = distance1
                    min_t = t
                if distance2 < min_dist2:
                    min_dist2 = distance2
            
            print(f"最小距离1: {min_dist1:.2f} 米 (在t={min_t:.2f}秒)")
            print(f"最小距离2: {min_dist2:.2f} 米")


def main():
    """主函数"""
    optimizer = CorrectedPhysicsOptimizer()
    optimizer.test_physics_understanding()


if __name__ == "__main__":
    main()
