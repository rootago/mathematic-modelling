"""
完全按照MATLAB代码逻辑的Python版本
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExactMatlabCopy:
    """完全按照MATLAB代码逻辑的版本"""
    
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
        
    def generate_points(self):
        """完全按照MATLAB的generate_points函数"""
        points = []
        # O1
        points.append([0, 193, 10])
        # 其他点都被注释掉了
        return np.array(points)
    
    def get_time_exact_matlab(self, t1, t2, v, theta, APos0, BPos0):
        """完全按照MATLAB的getTime函数"""
        # 参数
        AVel = 300
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        g = 9.8
        
        # 遍历所有点
        points = self.generate_points()
        times = []
        
        for i in range(len(points)):
            OPos = points[i]
            print(f'采点位置[{i+1}]：[{OPos[0]:.1f}, {OPos[1]:.1f}, {OPos[2]:.1f}]')
            
            # 仿真模拟
            t_step = 0.1
            t = 0
            flag1 = 0
            flag2 = 0
            current_t_start = 0
            current_t_end = 0
            
            while t <= 20:
                # 计算导弹位置
                APos = APos0 + (t + t1 + t2) * ADir * AVel
                
                # 计算烟幕弹位置
                DPos = np.array([
                    BPos0[0] + v * (t1 + t2) * np.cos(theta),
                    BPos0[1] + v * (t1 + t2) * np.sin(theta),
                    BPos0[2] - 0.5 * g * t2**2 - 3 * t
                ])
                
                # 计算距离
                OA = APos - OPos
                OD = DPos - OPos
                distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
                distance2 = np.linalg.norm(APos - DPos)
                
                if flag1 == 0 and distance1 <= 10:
                    flag1 = 1
                    current_t_start = t + t1 + t2
                    print(f'  开始遮蔽: t={t:.1f}, distance1={distance1:.2f}')
                    
                if flag1 == 1 and distance1 > 10:
                    current_t_end = t + t1 + t2
                    print(f'  结束遮蔽: t={t:.1f}, distance1={distance1:.2f}')
                    break
                
                if distance2 <= 10:
                    flag2 = 1
                    print(f'  距离2遮蔽开始: t={t:.1f}, distance2={distance2:.2f}')
                    
                if flag2 == 1 and distance2 > 10:
                    current_t_end = t + t1 + t2
                    print(f'  距离2遮蔽结束: t={t:.1f}, distance2={distance2:.2f}')
                    break
                
                # 手动更新时间
                t = t + t_step
            
            current_time = [current_t_start, current_t_end]
            print(f'current_time：[{current_time[0]:.2f}, {current_time[1]:.2f}]')
            times.append(current_time)
        
        times = np.array(times)
        t_start = np.max(times[:, 0])
        t_end = np.min(times[:, 1])
        
        print(f'最终结果: t_start={t_start:.2f}, t_end={t_end:.2f}')
        
        return t_start, t_end
    
    def test_exact_matlab_logic(self):
        """测试完全按照MATLAB逻辑的计算"""
        print("=== 测试完全按照MATLAB逻辑 ===")
        
        # 使用MATLAB中的参数
        t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
        APos0 = self.APos0[0]
        BPos0 = self.BPos0[0]
        
        print(f"参数: t1={t1}, t2={t2}, v={v}, theta={theta}")
        print(f"导弹初始位置: {APos0}")
        print(f"无人机初始位置: {BPos0}")
        
        t_start, t_end = self.get_time_exact_matlab(t1, t2, v, theta, APos0, BPos0)
        
        print(f"\n结果: t_start={t_start:.2f}, t_end={t_end:.2f}")
        
        if t_start < t_end:
            print(f"有效遮蔽时间: {t_end - t_start:.2f} 秒")
        else:
            print("无有效遮蔽时间")
    
    def test_different_parameters(self):
        """测试不同参数"""
        print("\n=== 测试不同参数 ===")
        
        test_cases = [
            (0.5, 1.0, 200, np.pi/6),
            (1.0, 2.0, 150, np.pi/4),
            (2.0, 3.0, 100, np.pi/3),
            (0.1, 0.5, 300, 0),
            (3.0, 4.0, 80, np.pi/2),
        ]
        
        for i, (t1, t2, v, theta) in enumerate(test_cases):
            print(f"\n--- 测试案例 {i+1}: t1={t1}, t2={t2}, v={v}, theta={theta:.3f} ---")
            
            for missile_idx in range(3):
                print(f"\n导弹 {missile_idx + 1}:")
                APos0 = self.APos0[missile_idx]
                BPos0 = self.BPos0[0]
                
                t_start, t_end = self.get_time_exact_matlab(t1, t2, v, theta, APos0, BPos0)
                
                if t_start < t_end:
                    print(f"✓ 有效遮蔽: [{t_start:.2f}, {t_end:.2f}]")
                else:
                    print("✗ 无有效遮蔽")


def main():
    """主函数"""
    optimizer = ExactMatlabCopy()
    
    # 测试完全按照MATLAB逻辑
    optimizer.test_exact_matlab_logic()
    
    # 测试不同参数
    optimizer.test_different_parameters()


if __name__ == "__main__":
    main()
