clc; clear;

% 微分方程 dy/dx = y
f = @(x,y) [y(2); y(2)+2*y(1)];

% 区间与初值
x0 = 0;
x_end = 2;
y0 = [1; 0]; 
h = 0.1;

% 向前差分
[x_forward, y_forward] = finite_difference_ode(f, x0, x_end, y0, h, 'forward');

% 向后差分
[x_backward, y_backward] = finite_difference_ode(f, x0, x_end, y0, h, 'backward');

% 输出结果
disp('x: ');
disp(x_forward)
disp('Forward差分 y:')
disp(y_forward(1,:)')

disp('Backward差分 y:')
disp(y_backward(1,:)')

