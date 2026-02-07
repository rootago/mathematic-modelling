% 数据
x = [0 1 2 3 4]';
y = [1.2 2.8 7.1 20.1 54.8]';

% 非线性模型 y = a*exp(b*x) + c
model = @(p, x) p(1)*exp(p(2)*x) + p(3);

% 初始猜测
p0 = [1 1 1];

% 拟合
params = my_least_squares_LM(model, x, y, p0);

% 拆解参数
a = params(1);
b = params(2);
c = params(3);

fprintf('拟合参数: a=%.4f, b=%.4f, c=%.4f\n', a, b, c);
