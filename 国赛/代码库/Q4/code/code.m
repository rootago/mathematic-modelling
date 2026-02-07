clear;clc
% 初始点
% x0 = [1.5,3.6,120,pi,1.5,3.6,120,pi,1.5,3.6,120,pi];
% x0 = [1.9483,2.9817,101.9706,3.1414,3.0317,3.7172,86.8369,3.1211,2.2835,4.0358,92.9330,3.1053];
x0 = [0.713,5.546,140,0.12,10.106,15.170,140,4.39,53.703,60.405,137.213,5/6*pi];

% 变量边界
% lb = [0, 0, 139, 0, 3, 6.8, 140, 4.3, 37, 10, 135, 2.5];
lb = [0, 0, 70, 0, 3, 3, 100, 3.2, 30, 10, 135, 2.5];
ub = [7, 7, 140, pi, 20, 20, 140, 1.5*pi, 40, 20, 140, pi];
% ub = [1, 1, 140, 0.2, 3.2, 7, 140, 4.4, 38, 12, 140, 2.7];
nvars = 12;

% 种群大小
PopSize = 200;

% 创建初始种群
InitialPopulation = zeros(PopSize, nvars);
x0_bounded = max(lb, min(ub, x0));
InitialPopulation(1, :) = x0_bounded;

% 其余个体在可行域内随机生成
for i = 2:PopSize
    for j = 1:nvars
        InitialPopulation(i, j) = lb(j) + (ub(j) - lb(j)) * rand();
    end
end

% 设置选项
options = optimoptions('ga', ...
    'Display','iter', ... % 显示每代信息
    'PopulationSize', PopSize, ... % 种群规模
    'InitialPopulationMatrix', InitialPopulation, ... % 使用自定义初始种群
    'MaxGenerations', 200, ... % 最大迭代代数
    'MutationFcn', {@mutationadaptfeasible, 0.9},...
    'FunctionTolerance', 1e-30, ... % 收敛容差
    'UseParallel', true ... % 如果有并行工具箱，可以开
);

[x,fval] = ga(@fun, nvars, [], [], [], [], lb, ub, [], options);
% [x,fval] = simulannealbnd(@fun, x0, lb, ub);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(-fval);

% 计算时间区间
final_result = -fun(x);
