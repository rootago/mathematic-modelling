import numpy as np

years = np.array([i for i in range(2000,2022)])
years = (years - 2000) / (2022-2000)  ## batch—normalize
## 年份 2000-2021

prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])
prices = (prices-prices.min()) / (prices.max()-prices.min())
pass

epoch = 10000
k = 1
b = 1
alpha = 0.01
for e in range(epoch):
    for x,label in zip(years,prices):
        pre = k*x + b # y_hat 
        loss = (pre - label) ** 2

        dk =  2*(pre-label)*x
        db =  2*(pre-label)

        k = k - alpha * dk
        b = b - alpha * db  

print(f"y={k}x+{b}")  ##梯度爆炸，被迫调小学习率 -> 所以就要数据规划，标准化，正则化等

print(f"{loss}")



