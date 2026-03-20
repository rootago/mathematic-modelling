import numpy as np
from tools import *
from pro2 import find_collision

const.PITCH = 55.0 # 初始值

def target_func(pitch):
    const.PITCH = pitch
    const.INIT_POS = 880 / pitch # 880cm对应16圈，计算初始位置
    collision_time, _ = find_collision()
    current_pos = theta2xy(t2theta(collision_time))
    return current_pos.x ** 2 + current_pos.y ** 2
if __name__ == "__main__":
    res = binary_search(30, 55, 450**2, target_func, return_mode = 'le')
    print(f"Optimal pitch: {res:.4f} cm")
    const.PITCH = res
    print(f"target_func result: {np.sqrt(target_func(const.PITCH)):.4f}")
    const.PITCH = 55.0 # 恢复初始值
