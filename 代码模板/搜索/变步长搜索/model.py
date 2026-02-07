import numpy as np

def variable_step_search(func, x0, h0=1.0, alpha=1.2, beta=0.5, 
                         tol=1e-6, max_iter=1000, directions=None):
    """
    通用变步长搜索算法 (Direct Search with Variable Step Size)
    
    Parameters:
    ----------
    func : callable
        目标函数 f(x)，输入 x (numpy array)，返回标量
    x0 : array-like
        初始点
    h0 : float
        初始步长
    alpha : float
        步长放大因子 (>1)
    beta : float
        步长缩小因子 (0<beta<1)
    tol : float
        步长收敛阈值
    max_iter : int
        最大迭代次数
    directions : list of np.array
        搜索方向（默认为单位坐标方向和其相反方向）
    
    Returns:
    -------
    x_best : ndarray
        最优解
    f_best : float
        最优目标值
    history : list
        迭代记录 (x, f(x), h)
    """
    
    x = np.array(x0, dtype=float)
    h = h0
    history = [(x.copy(), func(x), h)]
    
    n = len(x)
    if directions is None:
        # 默认用 ±坐标方向
        directions = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            directions.append(e)
            directions.append(-e)
    
    for k in range(max_iter):
        improved = False
        f_current = func(x)
        
        for d in directions:
            new_x = x + h * d
            f_new = func(new_x)
            if f_new < f_current:
                x = new_x
                h *= alpha  # 成功 → 扩大步长
                improved = True
                break
        
        if not improved:
            h *= beta  # 失败 → 缩小步长
        
        history.append((x.copy(), func(x), h))
        
        if h < tol:
            break
    
    return x, func(x), history

def objective(x):
        return (x[0]-2)**2 + (x[1]+3)**2   # 最优解在 (2, -3)

# 🔹 使用示例
if __name__ == "__main__":
    x0 = [0, 0]
    x_best, f_best, history = variable_step_search(objective, x0)

    print("最优解:", x_best)
    print("最优目标值:", f_best)
