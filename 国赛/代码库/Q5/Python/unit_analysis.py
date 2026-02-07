"""
单位分析 - 检查坐标尺度和距离计算
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_units():
    """分析单位问题"""
    print("=== 单位分析 ===")
    
    # 导弹初始位置
    APos0 = np.array([20000, 0, 2000])
    # 无人机初始位置  
    BPos0 = np.array([17800, 0, 1800])
    # 观测点
    OPos = np.array([0, 193, 10])
    
    print(f"导弹位置: {APos0} (单位: 米)")
    print(f"无人机位置: {BPos0} (单位: 米)")
    print(f"观测点位置: {OPos} (单位: 米)")
    
    # 计算初始距离
    distance_missile_obs = np.linalg.norm(APos0 - OPos)
    distance_drone_obs = np.linalg.norm(BPos0 - OPos)
    distance_missile_drone = np.linalg.norm(APos0 - BPos0)
    
    print(f"\n初始距离:")
    print(f"导弹到观测点: {distance_missile_obs:.2f} 米")
    print(f"无人机到观测点: {distance_drone_obs:.2f} 米") 
    print(f"导弹到无人机: {distance_missile_drone:.2f} 米")
    
    # 分析烟幕弹轨迹
    print(f"\n=== 烟幕弹轨迹分析 ===")
    
    t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
    
    print(f"参数: t1={t1}, t2={t2}, v={v}, theta={theta}")
    
    # 烟幕弹初始位置
    DPos_initial = np.array([
        BPos0[0] + v * (t1 + t2) * np.cos(theta),
        BPos0[1] + v * (t1 + t2) * np.sin(theta),
        BPos0[2] - 0.5 * 9.8 * t2**2
    ])
    
    print(f"烟幕弹初始位置: {DPos_initial}")
    print(f"烟幕弹到观测点距离: {np.linalg.norm(DPos_initial - OPos):.2f} 米")
    
    # 分析导弹轨迹
    print(f"\n=== 导弹轨迹分析 ===")
    
    alpha1 = np.arctan(APos0[2] / np.sqrt(APos0[0]**2 + APos0[1]**2))
    ADir = np.array([-np.cos(alpha1), 0, -np.sin(alpha1)])
    AVel = 300
    
    print(f"导弹方向角: {alpha1:.6f} 弧度 = {np.degrees(alpha1):.2f} 度")
    print(f"导弹方向向量: {ADir}")
    print(f"导弹速度: {AVel} 米/秒")
    
    # 导弹初始位置
    APos_initial = APos0 + (t1 + t2) * ADir * AVel
    print(f"导弹初始位置: {APos_initial}")
    print(f"导弹到观测点距离: {np.linalg.norm(APos_initial - OPos):.2f} 米")
    
    # 检查距离计算
    print(f"\n=== 距离计算检查 ===")
    
    OA = APos_initial - OPos
    OD = DPos_initial - OPos
    
    print(f"OA向量: {OA}")
    print(f"OD向量: {OD}")
    print(f"|OA|: {np.linalg.norm(OA):.2f}")
    print(f"|OD|: {np.linalg.norm(OD):.2f}")
    
    # 计算叉积
    cross_product = np.cross(OA, OD)
    print(f"OA × OD: {cross_product}")
    print(f"|OA × OD|: {np.linalg.norm(cross_product):.2f}")
    
    # 点到直线距离
    if np.linalg.norm(OA) > 1e-10:
        distance1 = np.linalg.norm(cross_product) / np.linalg.norm(OA)
        print(f"点到直线距离: {distance1:.2f} 米")
    else:
        print("距离1计算失败: |OA|太小")
    
    # 导弹到烟幕弹距离
    distance2 = np.linalg.norm(APos_initial - DPos_initial)
    print(f"导弹到烟幕弹距离: {distance2:.2f} 米")
    
    # 检查是否满足遮蔽条件
    print(f"\n=== 遮蔽条件检查 ===")
    print(f"距离1 <= 10: {distance1 <= 10 if 'distance1' in locals() else 'N/A'}")
    print(f"距离2 <= 10: {distance2 <= 10}")
    
    # 分析时间演化
    print(f"\n=== 时间演化分析 ===")
    
    times = np.arange(0, 21, 1)
    min_dist1 = np.inf
    min_dist2 = np.inf
    
    for t in times:
        # 导弹位置
        APos = APos0 + (t + t1 + t2) * ADir * AVel
        
        # 烟幕弹位置
        DPos = np.array([
            BPos0[0] + v * (t1 + t2) * np.cos(theta),
            BPos0[1] + v * (t1 + t2) * np.sin(theta),
            BPos0[2] - 0.5 * 9.8 * t2**2 - 3 * t
        ])
        
        # 计算距离
        OA = APos - OPos
        OD = DPos - OPos
        
        if np.linalg.norm(OA) > 1e-10:
            dist1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
            min_dist1 = min(min_dist1, dist1)
        
        dist2 = np.linalg.norm(APos - DPos)
        min_dist2 = min(min_dist2, dist2)
        
        if t <= 5:  # 只打印前几个时间点
            print(f"t={t:2.0f}: 导弹={APos}, 烟幕弹={DPos}, 距离1={dist1:.1f}, 距离2={dist2:.1f}")
    
    print(f"\n最小距离1: {min_dist1:.2f} 米")
    print(f"最小距离2: {min_dist2:.2f} 米")
    
    # 检查是否有任何时刻满足条件
    print(f"\n=== 详细时间步分析 ===")
    
    for t in np.arange(0, 20.1, 0.1):
        APos = APos0 + (t + t1 + t2) * ADir * AVel
        DPos = np.array([
            BPos0[0] + v * (t1 + t2) * np.cos(theta),
            BPos0[1] + v * (t1 + t2) * np.sin(theta),
            BPos0[2] - 0.5 * 9.8 * t2**2 - 3 * t
        ])
        
        OA = APos - OPos
        OD = DPos - OPos
        
        if np.linalg.norm(OA) > 1e-10:
            dist1 = np.linalg.norm(np.cross(OA, OD)) / np.linalg.norm(OA)
        else:
            dist1 = np.inf
            
        dist2 = np.linalg.norm(APos - DPos)
        
        if dist1 <= 10 or dist2 <= 10:
            print(f"t={t:.1f}: 距离1={dist1:.2f}, 距离2={dist2:.2f} - 满足条件!")
            break
    else:
        print("在0-20秒内没有找到满足条件的时刻")


if __name__ == "__main__":
    analyze_units()
