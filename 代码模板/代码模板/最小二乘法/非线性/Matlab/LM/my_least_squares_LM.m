function params = my_least_squares_LM(func, xdata, ydata, p0)
% my_least_squares_LM - 非线性最小二乘拟合模板（LM算法）
%
% 输入:
%   func  - @(params, x) 模型函数
%   xdata - 自变量数据
%   ydata - 因变量数据
%   p0    - 初始猜测参数
%
% 输出:
%   params - 拟合得到的参数

    % 残差函数 r(p) = ydata - func(p, xdata)
    residual = @(p) ydata - func(p, xdata);

    % LM算法选项
    options = optimoptions('lsqnonlin', ...
                           'Algorithm','levenberg-marquardt', ...
                           'Display','off');

    % 调用 lsqnonlin 进行拟合
    params = lsqnonlin(residual, p0, [], [], options);
end
