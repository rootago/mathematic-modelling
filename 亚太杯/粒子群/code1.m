clear;clc
%% 求函数的最大值
x0 = 2;
A = [];b = [];
Aeq = [];beq = [];
x_lb = -3;
x_ub = 3;
[x,fval] = fmincon(@Obj_fun1,x0,A,b,Aeq,beq,x_lb,x_ub)
fval = -fval


clear;clc
%% 绘制函数图像
x = -3:0.01:3;
y = 11*sin(x) + 7 *cos(5*x);
figure(1)
plot(x,y,'b-');
title('y = 11sinx+7cos5x')
hold on

%% 粒子群 step1 初始化参数
n = 20; %% 粒子数量，粒子数量越大，准度越高，运算时间越长
narvs = 1; %% 变量个数（函数有几个自变量）

%% 以后会优化，这是算法研究初期的设计
c1 = 2; %% 个体学习因子
c2 = 2; %% 社会学习因子
w = 0.9;%% 惯性权重

K = 20 ;%% 迭代的次数 注意饱和次数调整，问题复杂时K适当放大
vmax = 1.2; %% 粒子最大速度，速度不可太大，保证两次迭代点想接近，取自变量可行域的10%-20%
x_lb = -3; %% x的下界 解空间 [-3,3] 可行域长达 6
x_ub = 3; %% x的上界 上下界保证x跳出定义域

%% 两个自变量
%% narvs = 2
%% x_lb = [-3 -3]; x_ub = [ 3 3 ]
%% vmax = [1.2 1.2]

%% step2：初始化粒子的位置与速度
x = zeros(n,narvs);
%% 随机生成n个坐标
for i = 1:narvs
    x(:,i) = x_lb(i) + (x_ub(i)-x_lb(i)) * rand(n,1); %% [ n x 1 ]
end
%% 随即初始化粒子的速度,[-vmax,vmax]
v = -vmax + 2*vmax.*rand(n,narvs); %% .* 矩阵行点乘法
%% [1,2].*[3,4;5,6;7,8] 注意是行乘行
%% [1,2]+[3,4;5,6;7,8]  行乘行

%% step3：适应度计算,适应度就是目标函数值
fit = zeros(n,1); %% 初始化：每个粒子的适应度是0
for i = 1:n %% 循环整个粒子群，计算每一个粒子的适应度
    fit(i) = Obj_fun1(x(i,:));  %% 传入第i个粒子的变量
end

%%  初始化每个粒子迄今为止的最佳位置 (n x narvs的向量) // x = zeros(n,narvs);
pbest = x; 
%% 找到适应度最大的那个粒子的下标
 %% == 运算，产生一个01形成的列向量，找到为1的那一行，这个ind值做索引再去找gbest
ind = find (fit==max(fit),1);
%% 定义所有粒子迄今为止找到的最佳位置 (1 x narvs的向量)
gbest = x(ind,:);

%% 在图上标出粒子位置用于演示
h = scatter(x,fit,80,'*r');  %% 80是大小

%% step4：开始迭代，速度与位置更新
fitnessbest = ones(K,1); %% 初始化每次迭代得到的最佳适应度
for d = 1:K %% 迭代了K次
    for i = 1:n %% 第 i 个粒子
        %% 更新速度
        v(i,:) = w * v(i,:)...
+ c1 * rand(1) * (pbest(i,:)-x(i,:))...
+ c2 * rand(1) * (gbest-x(i,:));
        %% 如果速度超过最大速度限制，就要进行调整
        for j = 1:narvs
            if v(i,j) < -vmax(j)
                v(i,j) = -vmax(j);
            elseif v(i,j) > vmax(j)
                v(i,j) = vmax(j);
            end
        end
        %% 更新粒子的位置
        x(i,:) = x(i,:) + v(i,:);
        %% 确保粒子不超出可行域
        for j = 1:narvs
            if x(i,j) < x_lb(j)
                x(i,j) = x_lb(j);
            elseif x(i,j) > x_ub(j)
                x(i,j) = x_ub(j);
            end
        end
        %% 重新计算适应度
        fit(i) = Obj_fun1(x(i,:));  %% 传入第i个粒子的变量
        if fit(i) > Obj_fun1(pbest(i,:)) %% 这个粒子更新最佳位置
            pbest(i,:) = x(i,:);
        end
        if Obj_fun1(pbest(i,:)) > Obj_fun1(gbest)
            gbest = pbest(i,:); %% 更新所有粒子的最佳位置
        end
    end
    fitnessbest(d) = Obj_fun1(gbest);
    pause(0.1)
    h.XData = x;
    h.YData = -fit;  %%改变粒子坐标
end
        
figure(2)
plot(fitnessbest) %% 绘制每次迭代最佳适应度的变化曲线
xlabel('迭代次数');
disp('最佳位置： ');disp(gbest);
disp('最优解是： ');disp(Obj_fun1(gbest))


        