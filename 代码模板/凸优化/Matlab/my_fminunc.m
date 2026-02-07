function [x_opt, fval] = my_fminunc(func, x0, grad, useQuasiNewton)
% my_fminunc - 光滑无约束优化模板
%
% 输入:
%   func          - 目标函数 @(x) ...
%   x0            - 初始点
%   grad          - 梯度函数 @(x) ... 可选，如果没有可以传 []
%   useQuasiNewton- 是否使用拟牛顿法 (true/false)
%
% 输出:
%   x_opt - 最优解
%   fval  - 最优函数值

    if nargin < 4
        useQuasiNewton = true;  % 默认使用拟牛顿法
    end

    % 设置选项
    if useQuasiNewton
        options = optimoptions('fminunc', ...
                               'Algorithm','quasi-newton', ...
                               'SpecifyObjectiveGradient', ~isempty(grad), ...
                               'Display','off');
    else
        options = optimoptions('fminunc', ...
                               'Algorithm','trust-region', ...
                               'SpecifyObjectiveGradient', ~isempty(grad), ...
                               'Display','off');
    end

    % 调用 fminunc
    if isempty(grad)
        [x_opt,fval] = fminunc(func, x0, options);
    else
        [x_opt,fval] = fminunc(@(x) deal(func(x), grad(x)), x0, options);
    end
end
