% 目标函数
f = @(x) (x(1)-2)^2 + (x(2)+3)^2;

% 梯度函数（可选）
grad = @(x) [2*(x(1)-2); 2*(x(2)+3)];

% 初始点
x0 = [0 0];

% 调用模板（使用拟牛顿法）
% true quasi-newton 拟牛顿法
% false trust-region 信赖域法
[x_opt, fval] = my_fminunc(f, x0, grad, true);

disp('最优解:')
disp(x_opt)
disp('最优函数值:')
disp(fval)
