clear;clc
% 初始点
% x0 = [1.5,3.6,120,pi];
% x0 = [3.165,6.940,140,4.38]
% x0 = [53.703,60.405,137.213,5/6*pi]
% % 变量边界
% lb = [1.4, 1.4, 70, 0];
% ub = [12, 12, 140, pi];
% lb = [30, 10, 135, 2.5];
% ub = [40, 20, 140, pi]

% FY4
% x0 = [3.165, 6.940, 140, 4.39162];
% lb = [0, 0, 100, 3.2413];
% ub = [10, 10, 140, 1.5*pi];

% FY5
% x0 = [14.011, 1 , 139, 3.058];
% lb = [13.011, 0 , 138, 2.058];
% ub = [15.011, 2 , 140, 4.058];

% lb = [0, 0, 100, 0.5*pi];
% ub = [40, 20, 140, pi];

% x0 = [1.5,3.6,120,pi];
% lb = [0, 0, 70, 0];
% ub = [12, 12, 140, pi];

nvars = 4;

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

% 设置粒子群优化选项     'PlotFcn', @pswplotbestf, ...  % 绘制收敛曲线
options = optimoptions('particleswarm', ...
    'Display','iter', ... % 显示每代信息
    'SwarmSize', SwarmSize, ... % 粒子群大小
    'MaxIterations', 10000, ... % 最大迭代次数
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

% 计算位置
BPos0 = [6000, -3000, 700];
t1 = x(1);
t2 = x(2); 
v = x(3);
theta = x(4);
g = 9.8;
% 计算烟幕弹投放点位置
t = t1;
DPos = [BPos0(1) + v*(t1+t2)*cos(theta),...
        BPos0(2) + v*(t1+t2)*sin(theta),...
        BPos0(3) - 0.5*g*t2^2 - 3*t];
disp("烟幕弹投放点位置");
disp(DPos);

% 计算烟幕弹起爆点位置
t = t1+t2;
DPos = [BPos0(1) + v*(t1+t2)*cos(theta),...
        BPos0(2) + v*(t1+t2)*sin(theta),...
        BPos0(3) - 0.5*g*t2^2 - 3*t];
disp("烟幕弹起爆点位置");
disp(DPos);

% 计算时间区间
final_result = -fun(x);
initial_result = fun(x0_bounded);

fprintf('最终结果: %.6f\n', final_result);
fprintf('初始点x0结果: %.6f\n', initial_result);