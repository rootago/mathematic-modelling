% 数据
x = [0 1 2 3 4]';
y = [1.2 2.8 7.1 20.1 54.8]';

% 非线性模型 y = a*exp(b*x) + c
model = @(p, x) p(1)*exp(p(2)*x) + p(3);

% 初始猜测
p0 = [1 1 1];

% 参数边界
lb = [0 0 -Inf];
ub = [Inf Inf Inf];

% 拟合
params = my_least_squares(model, x, y, p0, lb, ub);

disp('拟合参数:')
disp(params)
