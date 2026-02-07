clear;clc
% 初始点
x0 = [1.5,3.6,1.5,3.6,1.5,3.6,120,pi];

% 变量边界
lb = [0, 1.4, 0, 1.4, 0, 1.4, 70, 0];
ub = [7, 7, 7, 7, 7, 7, 140, pi];

nvars = 8;

% 确保x0在边界范围内
x0_bounded = max(lb, min(ub, x0));

% PSO选项 10
% options = optimoptions('particleswarm', ...
%     'Display','iter', ...
%     'PlotFcn','pswplotbestf', ...     
%     'SwarmSize', 200, ...              
%     'MaxIterations', 200, ...          
%     'MaxStallIterations', 200, ...     
%     'FunctionTolerance', 1e-12, ...    
%     'InertiaRange', [0.2, 0.9], ...    % 惯性权重范围
%     'SelfAdjustmentWeight', 2.0, ...   % 个体学习因子
%     'SocialAdjustmentWeight', 2.0);    % 群体学习因子

options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'PlotFcn','pswplotbestf', ...       
    'SwarmSize', 50, ...            
    'MaxIterations', 100, ...          
    'MaxStallIterations', 200, ...     
    'FunctionTolerance', 1e-12, ...    
    'InertiaRange', [0.2, 0.9], ...    % 惯性权重范围
    'SelfAdjustmentWeight', 2.0, ...   % 个体学习因子
    'SocialAdjustmentWeight', 2.0);    % 群体学习因子

% 创建包含好初值的初始群体
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