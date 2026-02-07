function params = my_least_squares(func, xdata, ydata, p0, lb, ub)
% my_least_squares - 非线性最小二乘拟合模板
%
% 输入:
%   func  - @(params, x) 模型函数
%   xdata - 自变量数据
%   ydata - 因变量数据
%   p0    - 初始猜测参数
%   lb    - 参数下界
%   ub    - 参数上界
%
% 输出:
%   params - 拟合得到的参数

    % 优化选项
    options = optimoptions('lsqcurvefit', 'Display', 'off');

    % 调用 lsqcurvefit 进行最小二乘拟合
    params = lsqcurvefit(func, p0, xdata, ydata, lb, ub, options);

end

