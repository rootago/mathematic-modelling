clear;clc
% 初始点
% x0 = [1.5,3.6,120,pi,1.5,3.6,120,pi,1.5,3.6,120,pi];
% x0 = [1.9483,2.9817,101.9706,3.1414,3.0317,3.7172,86.8369,3.1211,2.2835,4.0358,92.9330,3.1053];
x0 = [0.713,5.546,140,0.12,10.106,15.170,140,4.39,53.703,60.405,137.213,5/6*pi];

% 变量边界
% lb = [0, 0, 139, 0, 3, 6.8, 140, 4.3, 37, 10, 135, 2.5];
lb = [0, 0, 70, 0, 0, 0, 100, 3, 0, 0, 100, 0.5*pi];
ub = [7, 7, 140, pi, 10, 10, 140, 1.5*pi, 40, 20, 140, pi];
% ub = [1, 1, 140, 0.2, 3.2, 7, 140, 4.4, 38, 12, 140, 2.7];

nvars = 12;

% 粒子群大小
SwarmSize = 200;

% 创建初始粒子群矩阵
InitialSwarm = zeros(SwarmSize, nvars);

% 确保x0在边界范围内
x0_bounded = max(lb, min(ub, x0));

% 第一个粒子设为初始点x0
InitialSwarm(1, :) = x0_bounded;

% 其余粒子在可行域内随机生成
for i = 2:SwarmSize
    for j = 1:nvars
        InitialSwarm(i, j) = lb(j) + (ub(j) - lb(j)) * rand();
    end
end

% 设置粒子群优化选项 - 原始配置
options = optimoptions('particleswarm', ...
    'Display','iter', ... % 显示每代信息
    'SwarmSize', SwarmSize, ... % 粒子群大小
    'MaxIterations', 200, ... % 最大迭代次数
    'InitialSwarmMatrix', InitialSwarm, ... % 使用自定义初始粒子群
    'FunctionTolerance', 1e-30, ... % 收敛容差
    'UseParallel', true ... % 如果有并行工具箱，可以开启
);

% 使用粒子群优化
[x,fval] = particleswarm(@fun, nvars, lb, ub, options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(-fval);

% 计算时间区间
final_result = -fun(x);
initial_result = fun(x0_bounded);

fprintf('最终结果: %.6f\n', final_result);
fprintf('初始点x0结果: %.6f\n', initial_result);