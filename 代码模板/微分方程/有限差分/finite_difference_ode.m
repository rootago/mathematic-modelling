function [x, y] = finite_difference_ode(f, x0, x_end, y0, h, method)
%FINITE_DIFFERENCE_ODE 通用有限差分求解 ODE 系统
%
% 输入:
%   f      - @(x,Y) 微分方程组 dY/dx = f(x,Y)，Y 可为向量
%   x0     - 初始 x
%   x_end  - 结束 x
%   y0     - 初值向量 Y(x0)
%   h      - 步长
%   method - 差分方法: 'forward', 'backward', 'central'
%
% 输出:
%   x - 离散 x
%   y - 每列对应 x 点的 Y 值，行数 = 变量个数

    x = x0:h:x_end;
    n = length(y0);          % 变量个数
    N = length(x);
    y = zeros(n, N);
    y(:,1) = y0(:);          % 保证 y0 是列向量

    switch lower(method)
        case 'forward'  % 显式欧拉
            for i = 1:N-1
                y(:,i+1) = y(:,i) + h * f(x(i), y(:,i));
            end

        case 'backward' % 隐式欧拉（需要 fsolve）
            for i = 1:N-1
                func = @(Yp1) Yp1 - y(:,i) - h*f(x(i+1), Yp1);
                y(:,i+1) = fsolve(func, y(:,i)); % 用上一列作为初值
            end

        case 'central'  % 中心差分
            for i = 2:N-1
                y(:,i+1) = y(:,i-1) + 2*h*f(x(i), y(:,i));
            end

        otherwise
            error('method 必须是 forward, backward, 或 central');
    end
end
