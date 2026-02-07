%% 基于bp神经网络的预测代码BP
%% 相关代码由哔哩哔哩up主：保丽龙小叮当整理
%% 出现任何学术不端行为与本人无关，代码仅供参考学习，如有侵权请后台私信
%% 参考相关教材、网络博客和公开论文
clear
close all
clc
load X 
load Y
%% 读取数据
input=X;
output=Y;
%% 训练集、测试集
input_train=input(:,1:1000);     %训练输入
output_train=output(:,1:1000);   %训练输出
input_test=input(:,1001:end);   %测试输入
output_test=output(:,1001:end); %测试输出

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);
inputn_train=mapminmax('apply',input_train,inputps);
%% 构建BP神经网络
net=newff(inputn,outputn,8);

% 网络参数
net.trainParam.epochs=1000;         % 训练次数
net.trainParam.lr=0.01;                   % 学习速率
net.trainParam.goal=0.000001;        % 训练目标最小误差
% net.dividefcn='';
%% BP神经网络训练
net=train(net,inputn,outputn);
%% BP神经网络测试
an=sim(net,inputn_test); %用训练好的模型进行仿真 
test_simu=mapminmax('reverse',an,outputps); % 预测结果反归一化
error=test_simu-output_test;      %预测值和真实值的误差
%% BP神经网络训练
%%网络训练数据输出
bn=sim(net,inputn);
%网络训练输出反归一化
train_simu_bp=mapminmax('reverse',bn,outputps);
%% BP神经网络测试绘图
%%真实值与预测值误差比较
figure(1)
plot(output_test,'bo-')
hold on
plot(test_simu,'r*-')
legend('期望值','预测值')
xlabel('数据组数'),ylabel('值'),title('测试集预测值和期望值的误差对比'),set(gca,'fontsize',12)
%% BP神经网络训练绘图
figure(2)
plot(output_train,'bo-')
hold on
plot(train_simu_bp,'r*-')
legend('期望值','预测值')
xlabel('数据组数'),ylabel('值'),title('训练集预测值和期望值的误差对比'),set(gca,'fontsize',12)
%计算误差
[~,len]=size(output_test);
MAE1=sum(abs(error./output_test))/len;
MSE1=error*error'/len;
RMSE1=MSE1^(1/2);
disp(['-----------------------误差计算--------------------------'])
disp(['平均绝对误差MAE为：',num2str(MAE1)])
disp(['均方误差MSE为：       ',num2str(MSE1)])
disp(['均方根误差RMSE为：  ',num2str(RMSE1)])


