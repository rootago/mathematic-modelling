clear;clc

f = @(x) sin(x); % 定义被积函数
a = 0; % 积分下限
b = pi; % 积分上限

% 使用梯形法则
result_trap = numerical_integration(f, a, b, 'trapezoidal', 100);

% 使用辛普森法则
result_simp = numerical_integration(f, a, b, 'simpson', 100);

% 使用高斯求积法
result_gauss = numerical_integration(f, a, b, 'gaussian', 100);

% 使用龙贝格积分
result_romberg = numerical_integration(f, a, b, 'romberg', 10, 1e-6);

% 使用蒙特卡洛积分
result_mc = numerical_integration(f, a, b, 'montecarlo', 10000);

disp(['梯形法则数值积分结果：   ', num2str(result_trap,'%.4f')]);
disp(['辛普森法则数值积分结果：   ', num2str(result_simp,'%.4f')]);
disp(['高斯求积法数值积分结果：   ', num2str(result_gauss,'%.4f')]);
disp(['龙贝格积分数值积分结果：   ', num2str(result_romberg,'%.4f')]);
disp(['蒙特卡洛积分数值积分结果：   ', num2str(result_mc,'%.4f')]);