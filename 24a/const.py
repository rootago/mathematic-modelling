import scipy.io as sio
import types

SIZE = 223 # 板凳龙223节
HEAD_LENGTH = 341 # 龙头341cm
BODY_LENGTH = 220 # 龙身220cm
TAIL_LENGTH = 220 # 龙尾220cm
WIDTH = 30 # 板宽30cm
DIAMETER = 5.5 # 孔径5.5cm
DISTC = 27.5 # 圆心距离边缘27.5cm
END_TIME = 500 # 模拟总时长500s

#========= Problem 1 ===========#

PITCH = 44.9533 # 螺旋间距55cm
VELOCITY = 100.0 # 行进速度100cm/s
INIT_POS = 16.0 # 初始16圈处

def Foo():
    print(PITCH)

# ========== 导出为 .mat 文件 ==========
def export_constants_mat(filename="constants.mat", pattern="ALL_CAPS") -> None:
    """
    自动导出当前模块中定义的常量到 .mat 文件。
    
    参数:
        filename: 输出的 .mat 文件名
        pattern: 导出模式
            "ALL_CAPS": 导出所有全大写变量（推荐）
            "NOT_PRIVATE": 导出所有不以单下划线开头的变量
    """
    # 获取当前模块的全局变量字典
    global_vars = globals().copy()
    
    # 准备收集要导出的数据
    data = {}
    
    for name, value in global_vars.items():
        # 跳过内置特殊变量（如 __name__, __file__）
        if name.startswith('__') and name.endswith('__'):
            continue
        
        # 跳过导入的模块
        if isinstance(value, types.ModuleType):
            continue
        
        # 根据命名模式筛选
        if pattern == "ALL_CAPS":
            if name.isupper():  # 全大写
                data[name] = value
        elif pattern == "NOT_PRIVATE":
            if not name.startswith('_'):  # 不以单下划线开头
                data[name] = value
        else:
            # 自定义筛选函数可在此扩展
            pass
    
    # 导出到 .mat 文件
    if data:
        sio.savemat(filename, data)
        print(f"Exported {len(data)} constants to {filename}")
        print("Variables:", list(data.keys()))
    else:
        print("No constants found to export.")

if __name__ == "__main__":
    export_constants_mat()  # 默认使用 "ALL_CAPS"