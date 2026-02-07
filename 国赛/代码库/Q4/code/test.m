clear;clc
% 初始点
x0 = [0.713,5.546,140,0.12,10.106,15.170,140,4.39,53.703,60.405,137.213,5/6*pi];
% 变量边界
lb = [0, 0, 70, 0, 5, 5, 100, 1.125*pi, 10, 10, 130, 2.652];
% ub = [12, 12, 140, pi, 12, 20, 140, pi, 12, 20, 140, pi];
ub = [7, 7, 140, pi, 10, 10, 140, 1.5*pi, 40, 40, 140, 2.678];
nvars = 12;

% 种群参数
PopSize = 200;

% 创建初始种群矩阵
InitialPopulation = zeros(PopSize, nvars);

% 第一个个体设置为x0（确保在边界内）
x0_bounded = max(lb, min(ub, x0)); % 将x0限制在边界范围内
InitialPopulation(1, :) = x0_bounded;

% 其余个体随机生成
for i = 2:PopSize
    for j = 1:nvars
        InitialPopulation(i, j) = lb(j) + (ub(j) - lb(j)) * rand();
    end
end

% 设置选项
options = optimoptions('ga', ...
    'Display','iter', ... % 显示每代信息
    'PopulationSize', PopSize, ... % 种群规模
    'MaxGenerations', 200, ... % 最大迭代代数
    'InitialPopulationMatrix', InitialPopulation, ... % 自定义初始种群
    'MutationFcn', {@mutationadaptfeasible, 0.9},...
    'FunctionTolerance', 1e-30, ... % 收敛容差
    'UseParallel', true ... % 如果有并行工具箱，可以开启
);

% 运行遗传算法
[x,fval] = ga(@fun, nvars, [], [], [], [], lb, ub, [], options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(-fval);

% 计算最终结果和初始点结果对比
final_result = -fun(x);
initial_result = fun(x0_bounded);

fprintf('最终结果: %.6f\n', final_result);
fprintf('初始点结果: %.6f\n', initial_result);
fprintf('改进程度: %.6f\n', final_result - initial_result);