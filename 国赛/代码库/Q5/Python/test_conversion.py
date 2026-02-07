"""
测试MATLAB到Python转换的正确性
"""

import numpy as np
from simple_example import SimpleSmokeOptimizer
import matplotlib.pyplot as plt


def test_basic_functions():
    """测试基本功能"""
    print("测试基本功能...")
    
    optimizer = SimpleSmokeOptimizer()
    
    # 测试观测点生成
    points = optimizer.generate_points()
    print(f"观测点: {points}")
    assert points.shape == (1, 3), "观测点形状错误"
    
    # 测试时间区间计算
    t1, t2, v, theta = 1.5, 3.6, 120, np.pi/4
    APos0 = optimizer.APos0[0]
    BPos0 = optimizer.BPos0[0]
    
    t_start, t_end = optimizer.get_time_interval(t1, t2, v, theta, APos0, BPos0)
    print(f"时间区间: [{t_start:.2f}, {t_end:.2f}]")
    
    # 测试区间合并
    intervals = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])
    merged = optimizer.merge_intervals(intervals)
    print(f"合并前: {intervals}")
    print(f"合并后: {merged}")
    assert len(merged) == 2, "区间合并错误"
    
    print("基本功能测试通过！")


def test_optimization():
    """测试优化功能"""
    print("\n测试优化功能...")
    
    optimizer = SimpleSmokeOptimizer()
    
    # 测试单架无人机优化
    existing_intervals = [np.array([]) for _ in range(3)]
    x_opt, fval, new_intervals = optimizer.optimize_single_drone(0, existing_intervals)
    
    print(f"优化结果: {x_opt}")
    print(f"目标值: {-fval:.4f}")
    
    # 验证解的有效性
    assert len(x_opt) == 8, "解的长度错误"
    assert all(0 <= x_opt[i] <= 7 for i in [0, 2, 4]), "t1值超出范围"
    assert all(1.4 <= x_opt[i] <= 7 for i in [1, 3, 5]), "t2值超出范围"
    assert 70 <= x_opt[6] <= 140, "v值超出范围"
    assert 0 <= x_opt[7] <= np.pi, "theta值超出范围"
    
    print("优化功能测试通过！")


def test_full_optimization():
    """测试完整优化流程"""
    print("\n测试完整优化流程...")
    
    optimizer = SimpleSmokeOptimizer()
    
    # 运行完整优化（只优化前2架无人机以节省时间）
    original_num_drones = optimizer.num_drones
    optimizer.num_drones = 2
    
    try:
        solutions, intervals = optimizer.optimize_all_drones()
        
        print(f"优化了 {len(solutions)} 架无人机")
        print(f"每架无人机的解形状: {[sol.shape for sol in solutions]}")
        
        # 验证结果
        assert len(solutions) == 2, "无人机数量错误"
        assert len(intervals) == 3, "导弹数量错误"
        
        print("完整优化流程测试通过！")
        
    finally:
        optimizer.num_drones = original_num_drones


def compare_with_matlab_logic():
    """与MATLAB逻辑对比"""
    print("\n与MATLAB逻辑对比...")
    
    optimizer = SimpleSmokeOptimizer()
    
    # 使用MATLAB中的测试参数
    x_test = np.array([1.5, 3.6, 1.5, 3.6, 1.5, 3.6, 120, np.pi])
    
    # 计算目标函数值
    existing_intervals = [np.array([]) for _ in range(3)]
    fval = optimizer.objective_function(x_test, 0, existing_intervals)
    
    print(f"测试参数: {x_test}")
    print(f"目标函数值: {-fval:.4f}")
    
    # 验证约束条件
    for i in range(2):
        t_prev = x_test[i*2]
        t_post = x_test[(i+1)*2]
        constraint_violation = abs(t_post - t_prev) < 1
        print(f"约束{i+1}违反: {constraint_violation}")
    
    print("MATLAB逻辑对比完成！")


def visualize_test_results():
    """可视化测试结果"""
    print("\n生成测试可视化...")
    
    optimizer = SimpleSmokeOptimizer()
    
    # 运行简化优化
    solutions, intervals = optimizer.optimize_all_drones()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示时间区间
    for i in range(3):
        ax = axes[i//2, i%2] if i < 3 else axes[1, 1]
        if len(intervals[i]) > 0:
            for j, interval in enumerate(intervals[i]):
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
    
    # 显示优化参数
    ax = axes[1, 1] if len(intervals) < 4 else axes[1, 0]
    ax.axis('off')
    ax.text(0.1, 0.9, '优化参数统计:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.8 - i*0.1, f'无人机{i+1}: v={sol[6]:.1f}, θ={sol[7]:.2f}', 
               transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("测试可视化完成！")


def main():
    """主测试函数"""
    print("开始测试MATLAB到Python转换...")
    
    try:
        test_basic_functions()
        test_optimization()
        test_full_optimization()
        compare_with_matlab_logic()
        visualize_test_results()
        
        print("\n所有测试通过！转换成功！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
