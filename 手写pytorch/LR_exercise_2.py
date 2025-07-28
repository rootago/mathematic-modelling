# pytorch 中，dataset与dataloader分开
import random 
import numpy as np
class MyDataset:  ## 可迭代数据
    def __init__(self,X1_datas,X2_datas,y_datas,batch_size,shuffle=True):
        self.X1_datas = X1_datas   # 所有数据
        self.X2_datas = X2_datas   # 所有数据
        self.y_datas = y_datas
        self.batch_size = batch_size  # 批次大小
        self.shuffle = shuffle        # 是否打乱
        

    def __len__(self): 
        return len(self.X1_datas)

    
    def __iter__(self): ## 返回具有next的对象
        # if self.shuffle:  ## 打乱数据
        #     random.shuffle(self.all_datas)  ## 不恰当，修改了原来的数据集合
        ## 另外，如果数据集合非常大，那么就无法全部加载到内存中，那么就无法打乱数据，非常不好
        # 不应该去打乱数据本身，而应该去打乱索引，而且不影响原来的索引
        return dataloader(self)  ## 构造一个dataloader对象，迭代器，具有next方法

class dataloader():  ## 迭代器
    def __init__(self,dataset):  ## 哪个数据集的加载器  // 哪个可迭代数据类型的迭代器 
        self.dataset = dataset
        self.cursor = 0 # 游标，边界，对于迭代器来说的游标边界，相当于是迭代器的index，整数类型
        self.indexs = [i for i in range(len(self.dataset))]  ## 索引，打乱只需要在索引列表打乱，而且不需要打乱原始数据
        if self.dataset.shuffle==True: ## 是否打乱
            random.shuffle(self.indexs) ## 做到了不修改原始数据，且节约空间，至打乱索引表，改变对应关系

    def __next__(self): ## 此self：观察dataset的iter函数：return dataloader(self)
        ## 所以这一步其实是 next(dataloader(self))  // 所以__next__(self)的self就是一个dataloader，已经是一个迭代器了
        ## 返还给batch_data的数据根据我们的需要，是一个list存放dataset我们随机取出的数据，作为训练数据
        if self.cursor >= len(self.dataset): 
            raise StopIteration ## 迭代器中值错误
        index = self.indexs[self.cursor:self.cursor+self.dataset.batch_size]
        batch_data_x_1 = self.dataset.X1_datas[index]
        batch_data_x_2 = self.dataset.X2_datas[index]
        batch_data_y = self.dataset.y_datas[index]  
        batch_data = np.array([batch_data_x_1,batch_data_x_2,batch_data_y])
        self.cursor += self.dataset.batch_size
        return batch_data 
    
    def __iter__(self):
        return self ## 习惯：迭代器最好是可迭代的数据类型





if __name__ == '__main__':
    years = np.array([i for i in range(2000,2022)])
    years = (years - 2000) / (2022-2000)  ## batch—normalize
    floor = np.array([random.randint(1,6) for i in range(22)])
    print(floor)
    ## 年份 2000-2021
    #print(years)
    prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])
    max_1 = prices.max()
    min_1 = prices.min()
    prices = (prices-prices.min()) / (prices.max()-prices.min())
    #print(prices)
    batch_size = 5
    shuffle =True
    dataset = MyDataset(years,floor,prices,batch_size,shuffle)
    epoch = 10000
    k1 = 1 
    k2 = 1
    b = 1
    alpha = 0.01
    print('--------------------------------------------------------------------')
    for e in range(epoch):
        for year,floor,price in dataset: # 把对象放在for上时，会自动调用__iter__方法

            #print(f"X : {x_batch_data}")
            #print(f"Y : {y_batch_data}\n")
            # for x,y in batch_data:
            #     print(x)
            #     print(y)
            # for i in range(len(x_batch_data)):
            #     pre = k * x_batch_data[i] + b
            #     loss = (pre - y_batch_data[i]) ** 2
            #     dk =  2*(pre-y_batch_data[i])*x_batch_data[i]
            #     db =  2*(pre-y_batch_data[i])
            #     k = k - alpha * dk
            #     b = b - alpha * db

            pre = k1 * year + k2 * floor + b
            loss = (pre - price) ** 2
            dk1 = (pre - price) * year
            dk2 = (pre - price) * floor
            db = (pre - price)

            k1 -= np.sum(dk1)/(batch_size-1) * alpha
            k2 -= np.sum(dk2)/(batch_size-1) * alpha
            b -= np.sum(db)/(batch_size-1) * alpha

print(f"y={k1}year+{k2}floor+{b}")  ##梯度爆炸，被迫调小学习率 -> 所以就要数据规划，标准化，正则化等

print(f"{np.sum(loss)}")     

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(years,prices)
# x = np.arange(0,1,0.001)
# y = k * x + b
# plt.plot(x,y)
# plt.show()
# print(max_1-min_1)
# print((k*(2025-2000)/22+b)*(max_1-min_1)+min_1)
# print(np.sum(loss))
