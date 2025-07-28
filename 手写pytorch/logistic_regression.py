import numpy as np

# [毛长，腿长]

dogs = np.array([[8.9,12],[9,11],[10,13],[9.9,11.2],[12.2,10.1],[9.8,13],[8.8,11.2]],dtype=np.float32)
cats = np.array([[3,4],[5,6],[3.5,5.5],[4.5,5.1],[3.4,4.1],[4.1,5.2],[4.4,4.4]], dtype = np.float32 )

labels = np.array([0]*7+[1]*7,dtype=np.int32).reshape(-1,1)
X = np.vstack((dogs,cats))  ## 垂直拼接


k = np.random.normal(0,1,(2,1)) ## 均值为0方差为1的正态分布
b = 0
def sigmoid(x):
    return 1/(1+np.exp(-x))

epoch = 1000
alpha = 0.01

for e in range(epoch):
    p = X @ k + b
    pre = sigmoid(p)

    loss = -np.mean(labels * np.log(pre) + (1-labels) * np.log(1-pre)) # 交叉熵损失,求平均防止梯度爆炸

    G = pre - labels  # 计算梯度 
    delta_k = X.T @ G
    delta_b = np.sum(G)
    k -= alpha * delta_k
    b -= alpha * delta_b

    print(loss)

print(f"k : {k}\n")
print(f"b : {b}\n") 

while True:
    hair = float(input("请输入毛长: "))
    leg = float(input("请输入腿长: "))
    x_test = np.array([hair, leg]).reshape(1,2) 
    p = sigmoid(x_test @ k + b)
    if p > 0.5: 
        print("猫")
    else: 
        print("狗")