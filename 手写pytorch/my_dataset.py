# Ctrl+/ 多行注释
# import random
# ##模拟dataset，数据集
# list1 =  [1,2,3,4,5,6,7]
# ##模拟dataloader 每两个两个去取，就是数据加载
# batch_size = 2 ## 从所有数据中一次拿出多少
# epoch = 10 ## 轮次，一遍学不会，多学几遍吧,学10次
# shuffle = True ## 打乱数据开关

# for i in range(epoch):
#     if shuffle: ## 随机打乱 每次学习都打乱
#         random.shuffle(list1)
#     print(list1)
#     for i in range(0,len(list1),batch_size):
#         batch_data = list1[i:i+batch_size]
#         print(batch_data) ## 以后就不是数字

# # 我希望每次学习的顺序不同，原本的内容（数据）打乱
# # 是否打乱取决于开发者目的


# 面向对象编程的数据集，复现pytorch
import random 
class MyDataset:   ## 还不是最终版，他是迭代器，不是可迭代数据类型
    def __init__(self,all_datas,batch_size,shuffle=True):  ## 初始化函数
        self.all_datas = all_datas    # 所有数据,传入的全体数据集
        self.batch_size = batch_size  # 批次大小，抽样大小
        self.shuffle = shuffle        # 是否打乱（开关）
        self.cursor = 0 # 游标，边界

    # python魔术方法，在某种特定场景下自动触发的方法
    ## 放在for循环那一瞬间触发，首次才触发，返回一个具有__next__的对象
    def __iter__(self):  ## for循环放上去触发，返回一个具有__next__的对象，只被出发一次，首次触发，后面的循环靠迭代器的迭代
        # print("iter now")
        if self.shuffle == True: ## 每次抽取只需打乱一次，反复打乱可能重复取出，不好
            random.shuffle(self.all_datas)
        # random.shuffle()用于将一个列表中的元素打乱顺序
        # 值得注意的是使用这个方法不会生成新的列表，只是将原列表的次序打乱。
        self.cursor = 0  # 重置游标，走到表的开头
        return self

    def __next__(self): ## for循环经过__iter__方法后，会调用__next__方法
        # i = 0 
        # print(f"next now {1}")
        # i += 1 
        if self.cursor >= len(self.all_datas): ## 判断游标是否已经走到表尾
            raise StopIteration ## 迭代器终止异常，跳出循环
        batch_data = self.all_datas[self.cursor:self.cursor+self.batch_size] # cursor - cursor + batach_size
        self.cursor += self.batch_size
        return batch_data 




if __name__ == '__main__':
    all_datas = [1,2,3,4,5,6,7,8,9,10]
    batch_size = 3
    shuffle =True
    dataset = MyDataset(all_datas,batch_size,shuffle)
    epoch = 2
    for e in range(epoch):
        for batch_data in dataset: # 把对象放在for上时，会自动调用__iter__方法
            print(batch_data)

