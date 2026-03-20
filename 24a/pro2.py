import numpy as np
from tools import *
def find_collision():
    begin, end, step = 0, const.END_TIME, 3.0
    collision_point = Coordinate()
    collision_time = 0
    while(step >= 1e-8):
        for t in np.arange(begin, end, step):
            thetas = t2fulltheta(t, 100)
            points = theta2xy(thetas)[:, -1] # 取出所有木板头把手的坐标
            collision_point = cal_moment_collision(points)
            if collision_point:
                collision_time = t
                begin = t - step
                end = t
                break
        step /= 10
    return collision_time, collision_point

if __name__ == "__main__":
    const.INIT_POS = 880 / const.PITCH # 880cm对应16圈，计算初始位置
    collision_time, collision_point = find_collision()
    print(f"Collision at time {collision_time} seconds: {collision_point}")
    head = theta2xy(t2theta(collision_time))
    print(f"Head Position: ({head.x:.4f}, {head.y:.4f}) cm")
    print(f"Distance from origin: {np.sqrt(head.x ** 2 + head.y ** 2) :.4f} cm")