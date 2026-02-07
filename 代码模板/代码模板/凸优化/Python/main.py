from scipy.optimize import minimize, differential_evolution
import numpy as np

def choose_optimizer(problem_type, func, x0, grad=None, bounds=None, constraints=None):
    """
    简单的优化算法选择器
    
    Parameters:
    - problem_type: 问题类型
    - func: 目标函数
    - x0: 初始点
    - grad: 梯度函数（可选）
    - bounds: 变量界限（可选）
    - constraints: 约束条件（可选）
    """
    
    if problem_type == "smooth_unconstrained":
        # 光滑无约束 → BFGS
        return minimize(func, x0, method='BFGS', jac=grad)
    
    elif problem_type == "smooth_with_bounds":
        # 有界约束 → L-BFGS-B
        return minimize(func, x0, method='L-BFGS-B', jac=grad, bounds=bounds)
    
    elif problem_type == "constrained":
        # 有约束 → SLSQP
        return minimize(func, x0, method='SLSQP', jac=grad, 
                       bounds=bounds, constraints=constraints)
    
    elif problem_type == "nonsmooth":
        # 不光滑 → Nelder-Mead（不需要梯度）
        return minimize(func, x0, method='Nelder-Mead')
    
    elif problem_type == "global_optimization":
        # 全局优化 → 进化算法
        return differential_evolution(func, bounds)
    
    else:
        # 默认用BFGS
        return minimize(func, x0, method='BFGS', jac=grad)


# 使用示例
if __name__ == "__main__":
    
    # 示例函数
    def rosenbrock(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def rosenbrock_grad(x):
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])
    
    x0 = np.array([0, 0])
    
    # 1. 无约束优化
    result1 = choose_optimizer("smooth_unconstrained", rosenbrock, x0, rosenbrock_grad)
    print(f"无约束优化: x={result1.x}, f={result1.fun}")
    
    # 2. 有界约束优化  
    bounds = [(-2, 2), (-2, 2)]
    result2 = choose_optimizer("smooth_with_bounds", rosenbrock, x0, rosenbrock_grad, bounds)
    print(f"有界约束优化: x={result2.x}, f={result2.fun}")
    
    # 3. 非光滑优化
    def abs_func(x):
        return np.sum(np.abs(x))
    
    result3 = choose_optimizer("nonsmooth", abs_func, np.array([1, -1]))
    print(f"非光滑优化: x={result3.x}, f={result3.fun}")
    
    # 4. 全局优化
    def multimodal(x):
        return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)
    
    bounds = [(-5, 5), (-5, 5)]
    result4 = choose_optimizer("global_optimization", multimodal, None, bounds=bounds)
    print(f"全局优化: x={result4.x}, f={result4.fun}")