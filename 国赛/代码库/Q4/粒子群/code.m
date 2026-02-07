clear;clc
% 初始点
% x0 = [1.5,3.6,120,pi,1.5,3.6,120,pi,1.5,3.6,120,pi];
% x0 = [1.9483,2.9817,101.9706,3.1414,3.0317,3.7172,86.8369,3.1211,2.2835,4.0358,92.9330,3.1053];
% x0 = [0.713,5.546,140,0.12,10.106,15.170,140,4.39,53.703,60.405,137.213,5/6*pi];  % 原来的初值

% 调试版本的初值 - 三架无人机都有合理的参数
x0 = [3.0, 5.0, 105, pi, 3.0, 5.0, 105, pi, 3.0, 5.0, 105, pi];

% 变量边界
% lb = [0, 0, 139, 0, 3, 6.8, 140, 4.3, 37, 10, 135, 2.5];
% lb = [0, 0, 70, 0, 0, 0, 100, 3, 0, 0, 100, 0.5*pi];  % 原来的边界
% ub = [7, 7, 140, pi, 10, 10, 140, 1.5*pi, 40, 20, 140, pi];  % 原来的边界
% ub = [1, 1, 140, 0.2, 3.2, 7, 140, 4.4, 38, 12, 140, 2.7];

% 调试版本，确保所有无人机都有合理范围
lb = [0.5, 2.0, 70, 0, 0.5, 2.0, 70, 0, 0.5, 2.0, 70, 0];           % 统一下界
ub = [15.0, 20.0, 140, 2*pi, 15.0, 20.0, 140, 2*pi, 15.0, 20.0, 140, 2*pi]; % 统一上界

nvars = 12;

% 确保x0在边界范围内
x0_bounded = max(lb, min(ub, x0));

% 设置粒子群优化选项 - 使用Q5的优化参数
options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'SwarmSize', 200, ...              % 增大粒子群规模
    'MaxIterations', 500, ...          % 增加最大迭代次数  
    'MaxStallIterations', 100, ...     % 增加停滞迭代次数
    'FunctionTolerance', 1e-15, ...    % 更严格的函数容忍度
    'InertiaRange', [0.2, 0.9], ...    % 调整惯性权重范围
    'SelfAdjustmentWeight', 2.0, ...   % 个体学习因子
    'SocialAdjustmentWeight', 2.0);    % 群体学习因子

% 创建包含好初值的初始群体 - 使用Q5的初值处理方法
swarm_size = 200;
initial_swarm = repmat(x0_bounded, swarm_size, 1);
% 在初值基础上添加随机扰动
for i = 1:swarm_size
    for j = 1:nvars
        % 在边界内随机扰动 ±20%
        range = ub(j) - lb(j);
        perturbation = (rand - 0.5) * 0.4 * range;
        initial_swarm(i, j) = max(lb(j), min(ub(j), x0_bounded(j) + perturbation));
    end
end

% 更新PSO选项以使用初始群体
options_with_init = optimoptions(options, 'InitialSwarmMatrix', initial_swarm);

% 使用粒子群优化
[x,fval] = particleswarm(@fun, nvars, lb, ub, options_with_init);

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