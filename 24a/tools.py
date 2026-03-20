import const
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
from typing import Any
def Foo():
    print(const.PITCH)

class Coordinate:
    def __init__(self, x: float = None, y: float = None) -> None: # type: ignore
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        fmt = lambda v: f"{v:.4f}" if v is not None else "None"
        return f"Coordinate = ({fmt(self.x)}, {fmt(self.y)})"
    def __bool__(self) -> bool:
        return self.x is not None and self.y is not None

class Seg:
    def __init__(self, p1: Coordinate, p2: Coordinate) -> None:
        self.x = [p1.x, p2.x]
        self.y = [p1.y, p2.y]

dragon_body_line = plt.plot([], [], )[0]
bodys = [plt.plot([], [], color='b', marker = 'o', markersize = 0, lw = 0.5)[0] for _ in range(const.SIZE - 1)]
def init_spiral():
    plt.axis('equal')
    thetas = np.linspace(0, 50 * np.pi, 1000)
    x, y = zip(*[(c.x, c.y) for c in theta2xy(thetas)])
    spiral = plt.plot(x, y, linestyle = "dotted")
    # draw_plank(Coordinate(0, 0), Coordinate(50, np.sqrt(BODY_LENGTH**2 - 50**2)))
    return spiral

def draw_plank(handle1: Coordinate, handle2: Coordinate):
    if not handle1 or not handle2:
        return ([], [])
    x1, y1 = handle1.x, handle1.y
    x2, y2 = handle2.x, handle2.y
    HALF_WIDTH = const.WIDTH / 2
    dx = x2 - x1
    dy = y2 - y1
    l = np.sqrt(dx**2 + dy**2)
    x = [
        x1 - (const.DISTC * dx + 1 * HALF_WIDTH * dy) / l,
        x2 + (const.DISTC * dx - 1 * HALF_WIDTH * dy) / l, 
        x2 + (const.DISTC * dx + 1 * HALF_WIDTH * dy) / l,
        x1 - (const.DISTC * dx - 1 * HALF_WIDTH * dy) / l,
        x1 - (const.DISTC * dx + 1 * HALF_WIDTH * dy) / l
    ]
    y = [
        y1 + (1 * HALF_WIDTH * dx - const.DISTC * dy) / l,
        y2 + (1 * HALF_WIDTH * dx + const.DISTC * dy) / l,
        y2 + (-1 * HALF_WIDTH * dx + const.DISTC * dy) / l,
        y1 + (-1 * HALF_WIDTH * dx - const.DISTC * dy) / l,
        y1 + (1 * HALF_WIDTH * dx - const.DISTC * dy) / l
    ]
    # seg_x1 = x1 - (DISTC * dx + s * HALF_WIDTH * dy) / l
    # seg_x2 = x2 + (DISTC * dx - s * HALF_WIDTH * dy) / l
    # seg_y1 = y1 + (s * HALF_WIDTH * dx - DISTC * dy) / l
    # seg_y2 = y2 + (s * HALF_WIDTH * dx + DISTC * dy) / l
    return (x, y)

def update_dragon(frame):
    global dragon_body_line
    time = frame * 0.05
    # print(frame)
    thetas = t2fulltheta(time)
    handles = theta2xy(thetas)[:,-1]
    x, y = zip(*[(c.x, c.y) for c in handles])
    # dragon_body_line.set_data(x, y)
    upd_lines = []
    for i in range(const.SIZE - 1):
        bodys[i].set_data(*draw_plank(handles[i], handles[i + 1]))
    return [dragon_body_line] + upd_lines
def draw_dragon():
    anim=animation.FuncAnimation(plt.gcf(), update_dragon, frames=range(9600), init_func=init_spiral, repeat=True, interval=50)
    plt.show()
    return anim

def segment_intersection(seg1: Seg, seg2: Seg, eps=1e-9) -> Coordinate:
    """
    返回两条线段的交点信息。
    输出格式：[has_intersection, P1, P2]
    - has_intersection: bool，表示是否有交点（包括重合线段）
    - P1, P2: Coordinate 对象，表示重合部分的两个端点。若只有一个点，则 P1 == P2；若无交点，则均为空 Coordinate。
    """
    # 提取坐标
    A1x, A1y = seg1.x[0], seg1.y[0]
    A2x, A2y = seg1.x[1], seg1.y[1]
    B1x, B1y = seg2.x[0], seg2.y[0]
    B2x, B2y = seg2.x[1], seg2.y[1]

    # 方向向量
    vax = A2x - A1x
    vay = A2y - A1y
    vbx = B2x - B1x
    vby = B2y - B1y

    # 叉积
    cross = vax * vby - vay * vbx

    # 辅助函数：判断浮点数是否接近零
    def near_zero(x):
        return abs(x) <= eps

    # 非平行情况（唯一交点）
    if abs(cross) > eps:
        delta_x = B1x - A1x
        delta_y = B1y - A1y
        t = (delta_x * vby - delta_y * vbx) / cross   # 参数 t 对应线段 A
        u = (delta_x * vay - delta_y * vax) / cross   # 参数 u 对应线段 B

        if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
            # 交点坐标
            ix = A1x + t * vax
            iy = A1y + t * vay
            # 将 t/u 截断到 [0,1] 范围内以保证点严格在线段上（避免微小误差）
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            ix = A1x + t * vax
            iy = A1y + t * vay
            return Coordinate(ix, iy)
        else:
            return Coordinate()

    # 平行情况
    # 检查是否共线：判断 (B1-A1) 与 va 的叉积是否接近零
    delta_x = B1x - A1x
    delta_y = B1y - A1y
    cross_collinear = delta_x * vay - delta_y * vax
    if abs(cross_collinear) > eps:
        return Coordinate()   # 平行但不共线

    # --- 以下处理共线情况 ---
    # 辅助函数：判断点是否在线段上（已知共线）
    def point_on_segment(px, py, sx1, sy1, sx2, sy2):
        # 线段方向向量
        svx = sx2 - sx1
        svy = sy2 - sy1
        # 如果线段退化为点
        if near_zero(svx) and near_zero(svy):
            return near_zero(px - sx1) and near_zero(py - sy1)
        # 使用投影参数
        len2 = svx * svx + svy * svy
        t = ((px - sx1) * svx + (py - sy1) * svy) / len2
        return -eps <= t <= 1 + eps

    # 处理退化情况（点线段）
    A_is_point = near_zero(vax) and near_zero(vay)
    B_is_point = near_zero(vbx) and near_zero(vby)

    if A_is_point and B_is_point:
        # 两个都是点
        if near_zero(A1x - B1x) and near_zero(A1y - B1y):
            return Coordinate(A1x, A1y)
        else:
            return Coordinate()
    elif A_is_point:
        # A 是点，B 是线段
        if point_on_segment(A1x, A1y, B1x, B1y, B2x, B2y):
            return Coordinate(A1x, A1y)
        else:
            return Coordinate()
    elif B_is_point:
        # B 是点，A 是线段
        if point_on_segment(B1x, B1y, A1x, A1y, A2x, A2y):
            return Coordinate(B1x, B1y)
        else:
            return Coordinate()

    # --- 两个都是正常线段（非退化）且共线 ---
    # 以 A1 为原点，方向向量为 va（非零）
    len2 = vax * vax + vay * vay   # 一定 > eps，因为非退化
    # 计算投影参数
    t_A1 = 0.0
    t_A2 = 1.0
    t_B1 = ((B1x - A1x) * vax + (B1y - A1y) * vay) / len2
    t_B2 = ((B2x - A1x) * vax + (B2y - A1y) * vay) / len2

    # 区间排序
    a_min, a_max = min(t_A1, t_A2), max(t_A1, t_A2)
    b_min, b_max = min(t_B1, t_B2), max(t_B1, t_B2)

    # 求一维区间交集
    inter_min = max(a_min, b_min)
    inter_max = min(a_max, b_max)

    if inter_min <= inter_max + eps:
        # 存在重叠
        # 计算对应的点坐标
        p_low_x = A1x + inter_min * vax
        p_low_y = A1y + inter_min * vay
        p_high_x = A1x + inter_max * vax
        p_high_y = A1y + inter_max * vay

        # 如果区间端点非常接近，视为一个点
        if inter_max - inter_min <= eps:
            return Coordinate(p_low_x, p_low_y)
        else:
            return Coordinate(p_low_x, p_low_y)
    else:
        return Coordinate()

def t2theta(t):
    a = const.PITCH / (2 * np.pi)
    v = const.VELOCITY
    init_theta = const.INIT_POS * 2 * np.pi
    t = np.asarray(t, dtype=float)

    def F(x):
        return x * np.sqrt(1 + x**2) + np.arcsinh(x)

    def dF(x):
        return 2 * np.sqrt(1 + x**2)

    c = F(init_theta)
    target = 2 * v / a * t
    x0 = init_theta if t.ndim == 0 else np.full_like(t, init_theta, dtype=float)
    res = optimize.newton(lambda x: F(x) - c + target, x0, fprime=dF)
    return res.item() if t.ndim == 0 else res

def theta2thetap(theta, length):
    """
    求解阿基米德螺线上与给定点相距指定长度的另一点辐角。
    螺线方程: r = a * theta, 其中 a = pitch/(2*pi)
    
    参数:
        theta : float 或 array_like, 点P的辐角（弧度）
        length: float 或 array_like, 两点间直线距离
    
    返回:
        res : float 或 ndarray, 点Q的辐角（弧度），与theta同型
    """
    a = const.PITCH / (2 * np.pi)
    theta = np.asarray(theta)
    length = np.asarray(length)
    
    # 标量扩展：若length为标量而theta为数组，扩展至相同形状
    if length.ndim == 0 and theta.ndim > 0:
        length = np.full(theta.shape, length)
    
    # 检查形状是否一致
    if theta.shape != length.shape:
        raise ValueError("theta和length必须具有相同的形状或可广播")
    
    # 定义方程函数
    def F(phi, t, l):
        return t**2 + phi**2 - 2 * t * phi * np.cos(phi - t) - (l / a) ** 2
    
    # 预分配输出数组
    res = np.zeros_like(theta, dtype=float)
    
    # 对每个元素求解
    for idx, (t, l) in enumerate(np.broadcast(theta, length)):
        # 单变量方程
        def eq(phi):
            return F(phi, t, l)
        
        # 寻找包含根的区间 [low, high]（确保 low < high 且函数值异号）
        low = t + 1e-6           # 略大于t，避免t处函数值为负
        f_low = eq(low)
        
        if f_low > 0:
            # 根位于 [t, low] 之间
            high = low
            low = t
        else:
            # 向右搜索直到函数值变号
            step = 0.1
            high = low + step
            max_iter = 100
            for _ in range(max_iter):
                f_high = eq(high)
                if f_high >= 0:
                    break
                low = high
                high = high + step
                step *= 1.5       # 动态增加步长
            else:
                raise RuntimeError(f"无法找到包含根的区间：theta={t}, length={l}")
        
        # 在区间上使用二分法求根
        sol = optimize.root_scalar(eq, bracket=[low, high], method='bisect')
        if sol.converged:
            res.flat[idx] = sol.root
        else:
            raise RuntimeError(f"求根未收敛：theta={t}, length={l}")
    
    return res

def theta2xy(theta):
    """
    输入角度theta，输出坐标。
    """
    def __theta2xy(theta):
        """
        输入theta，输出坐标。
        """
        a = const.PITCH / (2 * np.pi)
        resx = a * theta * np.cos(theta)
        resy = a * theta * np.sin(theta)
        return Coordinate(resx, resy)
    res = np.vectorize(__theta2xy)(theta)
    if np.isscalar(theta):
        return res.item() # 返回单个Coordinate对象
    else:
        return res # 返回Coordinate对象的列表
    
def t2fulltheta(t, num = const.SIZE):
    """
    给定时间，直接得出此时刻前 num 个板凳的theta
    """
    t = np.asarray(t, dtype=float)
    head_thetas = t2theta(t)  # 每一时刻头部的theta
    ans_theta = np.array(head_thetas).reshape(1, -1) # 整个龙的theta矩阵，每一行对应一个位置，每一列对应一个时间点
    for i in range(num):
        if i == 0:
            length = [const.HEAD_LENGTH - const.DISTC * 2] * t.size
        elif i == const.SIZE - 1:
            length = [const.TAIL_LENGTH - const.DISTC * 2] * t.size
        else:
            length = [const.BODY_LENGTH - const.DISTC * 2] * t.size
        new_theta = theta2thetap(ans_theta[-1].tolist(), length)
        ans_theta = np.vstack((ans_theta, np.array(new_theta).flatten()))
    return ans_theta

def cal_moment_thetas(head_theta) -> np.ndarray:
    """
    输入头的偏角，计算所有木板的偏角。
    """
    cur_theta = head_theta
    res = [cur_theta]
    for i in range(const.SIZE):
        if i == 0:
            length = const.HEAD_LENGTH - const.DISTC * 2
        elif i == const.SIZE - 1:
            length = const.TAIL_LENGTH - const.DISTC * 2
        else:
            length = const.BODY_LENGTH - const.DISTC * 2
        # cur_theta = eng.theta2thetap(matlab.double([cur_theta]), matlab.double([length]))[0]
        res.append(cur_theta)
    return np.array(res)

def cal_segments(p1: Coordinate, p2: Coordinate, s: int) -> Seg:
    """
    输入两个把手的坐标，以及s的取值计算！！！对应木板侧边！！！的起点和终点坐标。
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    HALF_WIDTH = const.WIDTH / 2
    dx = x2 - x1
    dy = y2 - y1
    l = np.sqrt(dx**2 + dy**2)
    seg_x1 = x1 - (const.DISTC * dx + s * HALF_WIDTH * dy) / l
    seg_x2 = x2 + (const.DISTC * dx - s * HALF_WIDTH * dy) / l
    seg_y1 = y1 + (s * HALF_WIDTH * dx - const.DISTC * dy) / l
    seg_y2 = y2 + (s * HALF_WIDTH * dx + const.DISTC * dy) / l
    return Seg(Coordinate(seg_x1, seg_y1), Coordinate(seg_x2, seg_y2))

def cal_moment_collision(points: list[Coordinate]) -> Coordinate:
    """
    输入某一时刻所有木板头把手的坐标，判断是否发生碰撞。
    只判断前5个木板与其后的木板是否发生碰撞。
    """
    from tools import segment_intersection
    # 提取 x 和 y 用于后续需要数组的操作
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    
    seg_sp = list(map(lambda p1, p2: cal_segments(p1, p2, 1), points[:-1], points[1:])) # s == positive 1 时的侧边线段
    seg_sn = list(map(lambda p1, p2: cal_segments(p1, p2, -1), points[:-1], points[1:])) # s == negative 1 时的外边线段
    # 以上两句是 s 分别取 +1 和 -1 时的线段，我不能确定正负是否一定对应内外但是不影响结果

    num_segs = len(seg_sp)
    for i in range(min(5, num_segs)):
        for j in range(i + 2, num_segs):
            res = segment_intersection(seg_sp[i], seg_sp[j])
            if res: return res
            res = segment_intersection(seg_sp[i], seg_sn[j])
            if res: return res
            res = segment_intersection(seg_sn[i], seg_sp[j])
            if res: return res
            res = segment_intersection(seg_sn[i], seg_sn[j])
            if res: return res
    return Coordinate() # 没有碰撞返回一个空坐标

def binary_search(begin: float, end: float, target: Any, condition_func, eps = 1e-6, return_mode: str = "default"):
    valid_modes = {"default", "lt", "le", "ge", "gt"}
    if return_mode not in valid_modes:
        raise ValueError("return_mode must be one of: 'default', 'lt', 'le', 'ge', 'gt'")

    begin_value = condition_func(begin)
    end_value = condition_func(end)

    low_x, high_x = begin, end
    low_value, high_value = begin_value, end_value
    if low_value > high_value:
        low_x, high_x = high_x, low_x
        low_value, high_value = high_value, low_value

    if not (low_value <= target <= high_value):
        raise ValueError("target is not bracketed by condition_func(begin) and condition_func(end)")

    while abs(high_x - low_x) > eps:
        mid = (low_x + high_x) / 2
        mid_value = condition_func(mid)
        if mid_value >= target:
            high_x = mid
            high_value = mid_value
        else:
            low_x = mid
            low_value = mid_value

    if return_mode == "default":
        return (low_x + high_x) / 2
    if return_mode == "lt":
        return low_x
    if return_mode == "le":
        if low_value <= target:
            return low_x
        return high_x
    if return_mode == "ge":
        if high_value >= target:
            return high_x
        return low_x
    return high_x

if __name__ == "__main__":
    ani = draw_dragon()
    ani.save("dragon.mp4", writer='ffmpeg', fps=20)