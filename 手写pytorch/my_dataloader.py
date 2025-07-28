# pytorch 中，dataset与dataloader分开

import random 
import numpy as np
class MyDataset:  ## 可迭代数据
    def __init__(self,all_datas,batch_size,shuffle=True):
        self.all_datas = all_datas    # 所有数据
        self.batch_size = batch_size  # 批次大小
        self.shuffle = shuffle        # 是否打乱
        

    def __len__(self): 
        return len(self.all_datas)

    
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
        if self.cursor >= len(self.dataset.all_datas): 
            raise StopIteration ## 迭代器中值错误
        index = self.indexs[self.cursor:self.cursor+self.dataset.batch_size]
        batch_data = self.dataset.all_datas[index]  
        self.cursor += self.dataset.batch_size
        return batch_data 
    
    def __iter__(self):
        return self ## 习惯：迭代器最好是可迭代的数据类型




if __name__ == '__main__':
    np.set_printoptions(threshold=2,linewidth=1)
    all_datas = np.array([1,2,3,4,5,6,7,8,9,10])
    batch_size = 3
    shuffle =True
    this_dataset = MyDataset(all_datas,batch_size,shuffle)

    epoch = 2
    for e in range(epoch):
        for batch_data in this_dataset: # 把对象放在for上时，会自动调用__iter__方法
            print(batch_data)
