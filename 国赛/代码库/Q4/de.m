clear; clc

% 初始点
x0 = [1.5,3.6,120,pi,1.5,3.6,120,pi,1.5,3.6,120,pi];
% x0 = [1.9483,2.9817,101.9706,3.1414,3.0317,3.7172,86.8369,3.1211,2.2835,4.0358,92.9330,3.1053];
lb = [0, 0, 70, 0, 0, 0, 70, 0, 0, 0, 70, 0];
ub = [7, 7, 140, pi, 12, 12, 140, pi, 20, 20, 140, pi];

options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'SwarmSize', 100, ...           % 群体大小
    'MaxIterations', 1000, ...      % 最大迭代次数
    'FunctionTolerance', 1e-6, ...
    'UseParallel', true);

[x, fval] = particleswarm(@fun, 12, lb, ub, options);

disp('最优解：');
disp(x);
disp('最优目标值：');
disp(-fval);
-fun(x)