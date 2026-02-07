clear; clc

x0 = [1.5,3.6,1.5,3.6,1.5,3.6,120,pi];

% 变量边界
lb = [0, 0, 0, 0, 0, 0, 70, pi/2];
ub = [7, 7, 7, 7, 7, 7, 140, pi];
nvars = 8;
swarmSize = 200;

% 构造初始种群 (第一行是x0，其余随机生成)
InitialSwarmMatrix = repmat(lb,swarmSize,1) + ...
    rand(swarmSize,nvars).*repmat((ub-lb),swarmSize,1);
InitialSwarmMatrix(1,:) = x0;   % 把x0放到第一行

% 设置 PSO 选项
options = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'SwarmSize', swarmSize, ...
    'InitialSwarmMatrix', InitialSwarmMatrix, ... 
    'MaxIterations', 200, ...
    'MaxStallIterations', 200, ...
    'FunctionTolerance', 1e-12, ...
    'UseParallel', true ...
);

% 调用粒子群优化
[x,fval] = particleswarm(@fun, nvars, lb, ub, options);

% 输出结果
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(-fval);

% 验证
final_result = -fun(x);
disp('最终目标函数值（验证）：');
disp(final_result);
