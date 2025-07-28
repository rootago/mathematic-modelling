# solve: (x-2)^2 = 0
from tqdm import trange  ##进度条 读： T range
epoch = 100000 ## 走几次
alpha = 0.01 ## lr 学习率
x = 2.00001 ## 初值过大，梯度爆炸，凯明初始化resnet
label = 0
pre = (x-2) **2 
loss = (pre-label)**2

for e in trange(epoch):
    pre = (x-2) **2 
    dx = 2 * (pre-label) * (x-2)
    x = x - alpha * dx

print(x)

    






