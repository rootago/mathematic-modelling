"""
详细调试版本 - 找出距离计算问题
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class DetailedDebugOptimizer:
    """详细调试版本的烟幕优化器"""
    
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
        
    def analyze_geometry(self, t1: float, t2: float, v: float, theta: float,
                        APos0: np.ndarray, BPos0: np.ndarray):
        """分析几何关系"""
        print(f"\n=== 几何分析 ===")
        print(f"参数: t1={t1}, t2={t2}, v={v}, theta={theta}")
        print(f"导弹初始位置: {APos0}")
        print(f"无人机初始位置: {BPos0}")
        print(f"观测点: {self.observation_points[0]}")
        
        # 计算导弹方向
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        print(f"导弹方向角: {alpha1:.4f}")
        print(f"导弹方向向量: {ADir}")
        
        # 分析几个时间点的位置
        times = [0, 1, 2, 5, 10, 15, 20]
        
        print(f"\n时间点分析:")
        print(f"{'时间':<8} {'导弹位置':<25} {'烟幕弹位置':<25} {'距离1':<10} {'距离2':<10}")
        print("-" * 80)
        
        for t in times:
            # 导弹位置
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
            
            print(f"{t:<8.1f} {str(APos):<25} {str(DPos):<25} {distance1:<10.2f} {distance2:<10.2f}")
        
        # 检查是否有任何距离小于10的情况
        print(f"\n距离分析:")
        min_distance1 = np.inf
        min_distance2 = np.inf
        
        for t in np.arange(0, 20.1, 0.1):
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
                min_distance1 = min(min_distance1, distance1)
                
            distance2 = np.linalg.norm(APos - DPos)
            min_distance2 = min(min_distance2, distance2)
        
        print(f"最小距离1 (点到直线): {min_distance1:.2f}")
        print(f"最小距离2 (导弹到烟幕弹): {min_distance2:.2f}")
        
        if min_distance1 <= 10:
            print("✓ 距离1条件可能满足")
        else:
            print("✗ 距离1条件不满足")
            
        if min_distance2 <= 10:
            print("✓ 距离2条件可能满足")
        else:
            print("✗ 距离2条件不满足")
    
    def test_different_parameters(self):
        """测试不同参数组合"""
        print("=== 测试不同参数组合 ===")
        
        # 测试参数组合
        test_cases = [
            # (t1, t2, v, theta, 描述)
            (1.0, 2.0, 200, np.pi/6, "高速度，小角度"),
            (2.0, 3.0, 150, np.pi/4, "中等参数"),
            (0.5, 1.5, 300, np.pi/3, "极高速度"),
            (3.0, 4.0, 100, 0, "零角度"),
            (1.5, 2.5, 120, np.pi/2, "垂直角度"),
        ]
        
        for i, (t1, t2, v, theta, desc) in enumerate(test_cases):
            print(f"\n--- 测试案例 {i+1}: {desc} ---")
            
            for missile_idx in range(3):
                print(f"\n导弹 {missile_idx + 1}:")
                self.analyze_geometry(t1, t2, v, theta, 
                                    self.APos0[missile_idx], 
                                    self.BPos0[0])
    
    def check_matlab_logic(self):
        """检查MATLAB逻辑"""
        print("\n=== 检查MATLAB逻辑 ===")
        
        # 使用MATLAB中的参数
        t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
        APos0 = self.APos0[0]
        BPos0 = self.BPos0[0]
        
        print("使用MATLAB原始参数:")
        print(f"t1={t1}, t2={t2}, v={v}, theta={theta}")
        
        # 计算导弹方向
        alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
        ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
        
        print(f"alpha1 = {alpha1:.6f}")
        print(f"ADir = {ADir}")
        
        # 检查烟幕弹位置计算
        print(f"\n烟幕弹位置计算:")
        print(f"BPos0 = {BPos0}")
        print(f"v*(t1+t2)*cos(theta) = {v*(t1+t2)*np.cos(theta):.2f}")
        print(f"v*(t1+t2)*sin(theta) = {v*(t1+t2)*np.sin(theta):.2f}")
        print(f"0.5*g*t2^2 = {0.5*self.g*t2**2:.2f}")
        
        # 计算初始位置
        DPos_initial = np.array([
            BPos0[0] + v * (t1 + t2) * np.cos(theta),
            BPos0[1] + v * (t1 + t2) * np.sin(theta),
            BPos0[2] - 0.5 * self.g * t2**2
        ])
        
        print(f"烟幕弹初始位置: {DPos_initial}")
        
        # 检查距离计算
        OPos = self.observation_points[0]
        print(f"观测点: {OPos}")
        
        # 计算导弹初始位置
        APos_initial = APos0 + (t1 + t2) * ADir * self.AVel
        print(f"导弹初始位置: {APos_initial}")
        
        # 计算初始距离
        OA = APos_initial - OPos
        OD = DPos_initial - OPos
        
        print(f"OA = {OA}")
        print(f"OD = {OD}")
        print(f"|OA| = {np.linalg.norm(OA):.2f}")
        print(f"|OD| = {np.linalg.norm(OD):.2f}")
        
        if np.linalg.norm(OA) > 1e-10:
            distance1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
            print(f"距离1 (点到直线) = {distance1:.2f}")
        else:
            print("距离1计算失败: |OA|太小")
            
        distance2 = np.linalg.norm(APos_initial - DPos_initial)
        print(f"距离2 (导弹到烟幕弹) = {distance2:.2f}")


def main():
    """主调试函数"""
    optimizer = DetailedDebugOptimizer()
    
    # 检查MATLAB逻辑
    optimizer.check_matlab_logic()
    
    # 测试不同参数
    optimizer.test_different_parameters()


if __name__ == "__main__":
    main()
