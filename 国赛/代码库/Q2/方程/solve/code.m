% 初始点
x0 = [1.5,3.6,120,pi];

% 变量边界
lb = [0, 0, 70, pi/2];
ub = [4, 4, 140, pi];

% 设置选项
options = saoptimset('Display','iter','MaxIter',200,'MaxFunEvals',500);

% 调用模拟退火
[x,fval] = simulannealbnd(@fun,x0,lb,ub,options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(fval);
