import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def get_data(file = "上海二手房价.csv"):
    datas = pd.read_csv(file,names=["y","x1","x2","x3","x4","x5","x6"],skiprows=1)
    # 将数据列重命名为 y, x1 至 x6。
    # 跳过首行：忽略文件的第一行（通常是原始列标题）。

    y = datas["y"].values.reshape(-1,1)   
    # [1,2,3,4,5,6,7,8...] ( 一位数组 ) -> (n,) -> [[1],[2],[3],[4],[5],[6],[7],[8]] (n x 1)
    
    # datas["y"]是从 DataFrame datas 中提取名为 "y" 的列（目标变量列）结果是一个 pandas Series 对象（一维数据结构）
    # .values 将 pandas Series 转换为 NumPy 数组转换后得到一维数组（形状为 (n_samples,)，例如 [10, 20, 30]）
    #.reshape(-1, 1)
    # -1：表示该维度由 NumPy 自动计算（保留所有样本）
    # 1：表示将数组转换为单列
    # 作用：将一维数组重塑为二维列向量（形状 (n_samples, 1)）
    # 为什么要重塑
    # 大多数框架（如 scikit-learn）要求输入数据是二维结构：
    # 特征矩阵：(n_samples, n_features)
    # 目标变量：(n_samples, 1) 或 (n_samples,)
    # 一维数组可能在某些操作中被误认为行向量（如矩阵乘法），重塑明确表示为列向量
    # 确保与特征矩阵 X 维度对齐 

    X = datas[[f"x{i}" for i in range(1,7)]].values # (n x 6)
    # X · w = y 

    # z-score : 
    mean_y = np.mean(y)
    std_y = np.std(y)

    mean_X = np.mean(X,axis=0,keepdims = True) 
    std_X = np.std(X,axis=0,keepdims = True)
    # axis=0 表示沿着行的方向（垂直方向）计算
    # keepdims=True 表示保持原始数组的维度结构

    X = (X - mean_X) / std_X
    y = (y - mean_y) / std_y 

    return X,y,mean_y,std_y,mean_X,std_X

if __name__ == "__main__":
    X,y,mean_y,std_y,mean_X,std_X = get_data()
    #print(X)
    print(y)
    print(std_y)
    print(mean_y)
    k = np.random.random((6,1))  # 初值 随机生成 6 x 1 的矩阵，参数是一个元组
    b = 1
    epoch = 10000
    alpha = 0.01 ## 0.01 梯度爆炸

    for e in range(epoch):
        pre = X @ k + b # C = A * B
        loss = np.sum((pre - y) ** 2)/(len(X)-1)  ## 引发梯度爆炸一部

        G =  (pre - y)/len(X) #  dL/dC · λ(某个常系数，不影响求导)  常数系数防爆炸
        nabla_B = X.T @ G  # dL/dB = AT · dL/dC
        db = np.sum(G)
        k = k - alpha * nabla_B
        b = b - alpha * db
    print(f"k : {k}\n")
    print(f"loss : {loss}\n")

    while True:
        bedroom = int(input("bedroom : "))
        ting = int(input("ting : "))
        wei = int(input("wei : "))
        area = int(input("area : "))
        floor = int(input("floor : "))
        year = int(input("year : "))

        test_x = np.array(([bedroom,ting,wei,area,floor,year]-mean_X)/std_X)

        pre_y = test_x @ k + b 

        print(f"price : {pre_y*std_y+mean_y}\n")


