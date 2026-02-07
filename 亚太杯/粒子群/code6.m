%% 非对称学习因子

clear;clc
x1 = -15:1:15;
x2 = -15:1:15;
[x1,x2] = meshgrid(x1,x2);
y = x1.^2+x2.^2-x1.*x2-10*x1-4*x2+60;
mesh(x1,x2,y)
xlabel('x1');ylabel('x2');zlabel('y');
axis vis3d 
hold on

%% 粒子群
n = 30;
narvs = 2;
c1_ini =2.5;
c2_ini =1;
c1_fin =0.5;
c2_fin =2.25;
K = 100;
w = 0.9;
vmax = [6 6];
x_lb = [-15 -15];
x_ub = [15 15];
x = zeros(n,narvs);
for i = 1:narvs
    x(:,i) = x_lb(i) + (x_ub(i)-x_lb(i)) * rand(n,1); %% [ n x 1 ]
end
v = -vmax + 2*vmax.*rand(n,narvs);
fit = zeros(n,1);

for i = 1:n %% 循环整个粒子群，计算每一个粒子的适应度
    fit(i) = Obj_fun2(x(i,:));  %% 传入第i个粒子的变量
end
pbest = x; 

ind = find (fit==max(fit),1);

gbest = x(ind,:);

h = scatter3(x(:,1),x(:,2),fit,'*r'); 

fitnessbest = ones(K,1); %% 初始化每次迭代得到的最佳适应度
c1 = 0 ; c2 = 0;
for d = 1:K %% 迭代了K次

    %% 更新学习因子
    c1 = c1_ini +  (c1_fin - c1_ini) * d / K ;
    c2 = c2_ini +  (c2_fin - c2_ini) * d / K ;

    %% 结合线性递减惯性权重
    w = 0.9 - (0.9-0.4) * (d/K);

    for i = 1:n %% 第 i 个粒子

        %% 更新速度
        v(i,:) = w * v(i,:)...
+ c1 * rand(1) * (pbest(i,:)-x(i,:))...
+ c2 * rand(1) * (gbest-x(i,:));
        for j = 1:narvs
            if v(i,j) < -vmax(j)
                v(i,j) = -vmax(j);
            elseif v(i,j) > vmax(j)
                v(i,j) = vmax(j);
            end
        end
        x(i,:) = x(i,:) + v(i,:);
        for j = 1:narvs
            if x(i,j) < x_lb(j)
                x(i,j) = x_lb(j);
            elseif x(i,j) > x_ub(j)
                x(i,j) = x_ub(j);
            end
         end
         fit(i) = Obj_fun2(x(i,:));
         if fit(i) < Obj_fun2(pbest(i,:)) %% 这个粒子更新最佳位置
            pbest(i,:) = x(i,:);
         end
         if fit(i) < Obj_fun2(gbest)
            gbest = pbest(i,:); %% 更新所有粒子的最佳位置
         end
    end
    fitnessbest(d) = Obj_fun2(gbest);
    pause(0.1)
    h.XData = x(:,1);
    h.YData = x(:,2);
    h.ZData = fit;  %%改变粒子坐标
end

figure(2)
plot(fitnessbest)
xlabel('迭代次数')
disp('最佳位置： '); disp(gbest);
disp('最优值 ： '); disp(Obj_fun2(gbest));