clear; clc

% 初始点
x0 = [0.713,5.546,140,0.12,10.106,15.170,140,4.39,53.703,60.405,137.213,5/6*pi];
% 变量边界
lb = [0, 0, 70, 0, 10, 10, 70, 4.38, 10, 10, 70, 1/2*pi];
% ub = [12, 12, 140, pi, 12, 20, 140, pi, 12, 20, 140, pi];
ub = [7, 7, 140, pi, 10, 10, 140, 4.38, 40, 40, 140, pi];

% 设置模拟退火选项
% options = optimoptions('simulannealbnd', ...
%     'Display','iter', ...        % 显示每次迭代信息
%     'MaxIterations', 40000, ...
%     'FunctionTolerance', 1e-30);

options = optimoptions('simulannealbnd', ...
    'Display','iter', ...
    'MaxIterations', 40000, ...
    'FunctionTolerance', 1e-6, ...           % 放宽容差
    'InitialTemperature', 100, ...           % 设置较高初始温度
    'ReannealInterval', 100, ...             % 重新退火
    'TemperatureFcn', @temperatureexp, ...   % 尝试指数降温
    'AnnealingFcn', @annealingfast);         % 快速退火函数

% 调用模拟退火
[x,fval] = simulannealbnd(@fun, x0, lb, ub, options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(- fval);

% 计算时间区间
final_result = -fun(x);
