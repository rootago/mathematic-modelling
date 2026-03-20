# import matlab.engine
import numpy as np
from tools import *
# from openpyxl import Workbook
# eng = matlab.engine.start_matlab('-uft8')

t_out = np.array([0, 60, 120, 180, 240, 300])
pos_out = np.array([0, 1, 51, 101, 151, 201, const.SIZE])

# result = eng.p1t2theta(t_out)
# result2 = eng.p1theta2xy(result)
# result = np.array(result) / PI
# result2 = np.array(result2)
# lengths = [HEAD_LENGTH] + [BODY_LENGTH] * (SIZE - 2) + [TAIL_LENGTH]
# with np.printoptions(precision=4, suppress=True, formatter={'float': '{:10.4f}'.format}):
#     print("t    ", np.array(t_out)[0])
#     print("theta", result[0])
#     print("x    ", result2[0, :])
#     print("y    ", result2[1, :])

times = np.arange(0, const.END_TIME + 1, 1) # 每秒计算一次
ans_theta = t2fulltheta(times)  # 每一时刻头部的theta

# theta_sheet = Workbook().active
# for _ in range(ans_theta.shape[0]):
#     theta_sheet.append(ans_theta[_].tolist())
# theta_sheet.parent.save("theta.xlsx")

ans_coords = theta2xy(ans_theta)

np.set_printoptions(linewidth=200, precision=4, suppress=True)
# 问1的输出
print("t    ", t_out)
for i in range(len(pos_out)):
    # 提取指定位置和时间点的坐标数组
    coords_subset = ans_coords[pos_out[i]][t_out]   # 假设是对象数组
    
    # 使用 np.array2string 自定义格式
    s = np.array2string(
        coords_subset,
        formatter={'object': lambda c: f"({c.x:10.4f}, {c.y:10.4f})"},  # 控制坐标格式 #type: ignore
        separator=' ',          # 元素之间用空格分隔
        max_line_width=200,     # 足够大，避免换行
        threshold=len(coords_subset)  # 确保不省略
    )
    # 去掉数组自带的方括号（如果想去掉）
    s = s.strip('[]')
    print(f"pos_{pos_out[i]:3d}", s)

