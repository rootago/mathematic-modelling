% ode15i求解DAE问题的简单模板
% DAE形式：F(t,y,y') = 0

clear; clc;

%% 1. 定义DAE方程组
% F(t,y,y') = 0 的形式
function res = dae_equations(t, y, yp)
    % 示例：简单的DAE系统
    % y1' - y2 = 0
    % y1 + y2 - exp(-t) = 0
    
    res = zeros(2,1);
    res(1) = yp(1) - y(2);           % y1' - y2 = 0
    res(2) = y(1) + y(2) - exp(-t);  % y1 + y2 - exp(-t) = 0
end

%% 2. 设置初始条件和时间范围
t0 = 0;              % 初始时间
tf = 2;              % 结束时间
% tspan = [t0 tf];     % 时间范围
% 自定义时间点间距
tspan = linspace(t0, tf, 201);

y0 = [0.5; 0.5];     % [y1(0); y2(0)] - 状态变量的初值
yp0 = [0.5; -0.5];   % [y1'(0); y2'(0)] - 状态变量导数的初值
%% 3. 使用ode15i求解
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode15i(@dae_equations, tspan, y0, yp0, options);

%% 4. 显示所有时间点的结果
disp('求解完成！');
disp(['时间点数量: ', num2str(length(t))]);
disp(' ');

% 输出所有时间点的解
disp('所有时间点的数值解：');
disp('    时间t        y1          y2');
disp('--------------------------------');
for i = 1:length(t)
    fprintf('%8.4f    %10.6f    %10.6f\n', t(i), y(i,1), y(i,2));
end

disp(' ');
% 验证最终解是否满足代数约束
final_constraint = y(end,1) + y(end,2) - exp(-t(end));
disp(['最终时刻约束误差: ', num2str(abs(final_constraint))]);