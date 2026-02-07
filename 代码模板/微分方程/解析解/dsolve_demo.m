% 定义符号变量
syms x(t) y(t)   % 如果有多个未知函数就都写上

% 写出方程
equ1 = diff(x,t) == y;      % 示例: Dx = y
equ2 = diff(y,t) == -x;     % 示例: Dy = -x
equ  = [equ1, equ2];        % 把方程组成数组

% 初值条件
cond = [x(0) == 0, y(0) == 1];

% 求解
[xSol, ySol] = dsolve(equ, cond);

% （可选）简化结果
xSol = simplify(xSol);
ySol = simplify(ySol);

% （可选）绘图
fplot(xSol, [0,10]); hold on;
fplot(ySol, [0,10]);
legend('x(t)', 'y(t)');
