clear;clc
% 初始点
x0 = [1.5,3.6,120,pi];

% 变量边界
lb = [0, 0, 70, 0];
ub = [12, 12, 140, pi];

nvars = 4;
% 设置选项
options = optimoptions('ga', ...
    'Display','iter', ...
    'PlotFcn', @gaplotbestf, ...
    'PopulationSize', 200, ...
    'MaxGenerations', 100, ...
    'MaxStallGenerations', 200, ...
    'FunctionTolerance', 1e-12, ...
    'UseParallel', true);

% 调用遗传算法
[x,fval] = ga(@fun, nvars, [], [], [], [], lb, ub, [], options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(- fval);

% 时间区间
result = -fun(x)
