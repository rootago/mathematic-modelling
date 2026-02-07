clear;clc
% 初始点
x0 = [1.5,3.6,1.5,3.6,1.5,3.6,120,pi];

% 变量边界
lb = [0, 0, 0, 0, 0, 0, 70, pi/2];
ub = [7, 7, 7, 7, 7, 7, 140, pi];

nvars = 8;
% 设置选项
options = optimoptions('ga', ...
    'Display','iter', ...            % 显示每代信息
    'PopulationSize', 200, ...       % 种群规模
    'MaxGenerations', 200, ...      % 最大迭代代数
    'MaxStallGenerations', 200, ...  % 收敛前允许停滞的代数
    'MutationFcn', {@mutationuniform, 0.15},...
    'FunctionTolerance', 1e-12, ...  % 收敛容差
    'UseParallel', true ...          % 如果有并行工具箱，可以开
);

% 调用遗传算法
[x,fval] = ga(@fun, nvars, [], [], [], [], lb, ub, [], options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(- fval);

% 计算时间区间
final_result = -fun(x)
