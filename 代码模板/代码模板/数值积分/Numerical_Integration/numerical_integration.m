function result = numerical_integration(f, a, b, method, varargin)
    % numerical_integration: 通用数值积分函数
    % 
    % 输入参数:
    %   f: 被积函数句柄 (例如, @(x) sin(x))
    %   a: 积分下限
    %   b: 积分上限
    %   method: 积分方法，可选值:
    %       'trapezoidal' - 梯形法则
    %       'simpson' - 辛普森法则
    %       'simpson38' - 辛普森3/8法则
    %       'gaussian' - 高斯求积法
    %       'romberg' - 龙贝格积分
    %       'montecarlo' - 蒙特卡洛积分
    %   varargin: 可选参数，取决于方法:
    %       - 'trapezoidal', 'simpson', 'simpson38': n (子区间数)
    %       - 'gaussian': n (高斯节点数)
    %       - 'romberg': max_iter (最大迭代次数), tol (容差)
    %       - 'montecarlo': N (采样点数)
    %
    % 输出参数:
    %   result: 积分结果

    switch method
        case 'trapezoidal'
            % 梯形法则
            n = varargin{1};
            x = linspace(a, b, n+1);
            y = f(x);
            result = (b-a)/(2*n) * (y(1) + 2*sum(y(2:end-1)) + y(end));

        case 'simpson'
            % 辛普森法则
            n = varargin{1};
            if mod(n, 2) ~= 0
                error('辛普森法则需要偶数子区间数');
            end
            x = linspace(a, b, n+1);
            y = f(x);
            result = (b-a)/(3*n) * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));

        case 'simpson38'
            % 辛普森3/8法则
            n = varargin{1};
            if mod(n, 3) ~= 0
                error('辛普森3/8法则需要3的倍数子区间数');
            end
            x = linspace(a, b, n+1);
            y = f(x);
            result = (b-a)/(8*n) * (y(1) + 3*sum(y(2:3:end-2)) + 3*sum(y(3:3:end-1)) + 2*sum(y(4:3:end-3)) + y(end));

        case 'gaussian'
            % 高斯求积法
            n = varargin{1};
            [nodes, weights] = lgwt(n, a, b); % 调用lgwt函数获取高斯节点和权重
            result = sum(weights .* f(nodes));

        case 'romberg'
            % 龙贝格积分
            max_iter = varargin{1};
            tol = varargin{2};
            R = zeros(max_iter, max_iter);
            h = b - a;
            R(1,1) = (f(a) + f(b)) * h / 2;

            for k = 2:max_iter
                h = h / 2;
                R(k,1) = R(k-1,1)/2 + h * sum(f(a + (2*(1:2^(k-2))-1)*h));

                for m = 2:k
                    R(k,m) = (4^(m-1)*R(k,m-1) - R(k-1,m-1)) / (4^(m-1)-1);
                end

                if abs(R(k,k) - R(k-1,k-1)) < tol
                    result = R(k,k);
                    return;
                end
            end
            result = R(max_iter,max_iter);

        case 'montecarlo'
            % 蒙特卡洛积分
            N = varargin{1};
            x = a + (b-a) * rand(N, 1);
            result = (b-a) * mean(f(x));

        otherwise
            error('未知的积分方法');
    end
end

function [x, w] = lgwt(n, a, b)
    % lgwt: 计算高斯-勒让德节点和权重
    % 输入:
    %   n: 节点数
    %   a: 区间下限
    %   b: 区间上限
    % 输出:
    %   x: 节点
    %   w: 权重

    n = n-1;
    n1 = n+1;
    xu = linspace(-1, 1, n1)';
    y = cos((2*(0:n)'+1)*pi/(2*n+2)) + (0.27/n1)*sin(pi*xu*n/n1);
    L = zeros(n1, n1+1);
    y0 = 2;
    while max(abs(y-y0)) > eps
        L(:,1) = 1;
        L(:,2) = y;
        for k = 2:n1
            L(:,k+1) = ((2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1))/k;
        end
        Lp = n1*(L(:,n1)-y.*L(:,n1+1))./(1-y.^2);
        y0 = y;
        y = y0 - L(:,n1+1)./Lp;
    end
    x = (a*(1-y) + b*(1+y))/2;
    w = (b-a)./((1-y.^2).*Lp.^2)*(n1/n)^2;
end