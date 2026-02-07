% 定义符号变量
syms  y(x)   % 如果有多个未知函数就都写上

% 写出方程
equ1 = diff(y,x) == (f(x)-x*y^2)/x^2;

equ  = [equ1];  

% 求解
[ySol] = dsolve(equ)

% （可选）简化结果
ySol = simplify(ySol)
