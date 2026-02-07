%% 对结合自适应惯性权重和压缩因子法的粒子群算法进行效果测试

clear;clc

n = 1000;
narvs = 30 ;
K = 200; 
wmax = 0.9 ; wmin = 0.4;
c1 = 2.05; c2 = 2.05;
C = c1+c2;
phi = 2 / abs ( 2 - C- sqrt(C^2 - 4 * C)  ); 
x_lb = -100 *ones(1,30);
x_ub = 100 *ones(1,30);
vmax = 30 * ones(1,30); % 取vmax为可行域的15%
x = zeros(n,narvs);

for i = 1:narvs
    x(:,i) = x_lb(i) + (x_ub(i)-x_lb(i)) * rand(n,1);
end
v = -vmax + 2*vmax.*rand(n,narvs);

fit = zeros(n,1);
for i = 1:n %% 循环整个粒子群，计算每一个粒子的适应度
    fit(i) = Obj_fun3(x(i,:));  %% 传入第i个粒子的变量
end
pbest = x; 
ind = find (fit==max(fit),1);
gbest = x(ind,:);

fitnessbest = ones(K,1); %% 初始化每次迭代得到的最佳适应度

for d = 1:K %% 迭代了K次  
    for i = 1:n %% 第 i 个粒子

        f_i = fit(i);  % 取出第i个粒子的适应度
        f_avg = sum(fit)/n;  % 计算此时适应度的平均值
        f_min = min(fit); % 计算此时适应度的最小值
        if f_i <= f_avg  
            w = wmin + (wmax - wmin)*(f_i - f_min)/(f_avg - f_min);
        else
            w = wmax;
        end

        %% 更新速度
        v(i,:) = phi*(w * v(i,:)...
+ c1 * rand(1) * (pbest(i,:)-x(i,:))...
+ c2 * rand(1) * (gbest-x(i,:)));
        %% 更新位置
        x(i,:) = x(i,:) + v(i,:);

        for j = 1:narvs
            if x(i,j) < x_lb(j)
                x(i,j) = x_lb(j);
            elseif x(i,j) > x_ub(j)
                x(i,j) = x_ub(j);
            end
        end
        fit(i) = Obj_fun3(x(i,:));
        if fit(i) < Obj_fun3(pbest(i,:)) %% 这个粒子更新最佳位置
            pbest(i,:) = x(i,:);
        end
        if fit(i) < Obj_fun3(gbest)
            gbest = pbest(i,:); %% 更新所有粒子的最佳位置
        end
    end
    fitnessbest(d) = Obj_fun3(gbest);
end
figure(1)
plot(fitnessbest)
xlabel('迭代次数')
disp('最佳位置： '); disp(gbest);
disp('最优值 ： '); disp(Obj_fun3(gbest));