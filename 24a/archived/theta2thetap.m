function res = theta2thetap(theta, length)
    % theta2thetap - 求解阿基米德螺线上与给定点相距指定长度的另一点辐角
    %
    % 螺线方程: r = a * theta, 其中 a = 55/(2*pi)
    % 输入:
    %   theta  - 点P的辐角（弧度），可为标量或向量
    %   length - 两点间直线距离，可为标量或与theta同维的向量
    % 输出:
    %   res    - 点Q的辐角（弧度），与theta同维

    % 螺距常数
    PITCH = 55;
    a = PITCH / (2 * pi);

    % 若length为标量而theta为向量，进行标量扩展
    if isscalar(length) && ~isscalar(theta)
        length = repmat(length, size(theta));
    end

    % 预分配输出数组
    res = zeros(size(theta));

    % 对每个元素进行求解
    for i = 1:numel(theta)
        t = theta(i);       % 当前P点辐角
        l = length(i);      % 当前距离

        % 定义方程: F(phi) = 0，其中phi为待求辐角
        F = @(phi) t^2 + phi.^2 - 2*t*phi.*cos(phi - t) - (l/a)^2;

        % --- 寻找包含根的区间 [x_low, x_high] ---
        % 初始下界略大于t（避免t本身，因为t处F为负）
        x_low = t + 1e-6;
        f_low = F(x_low);

        if f_low > 0
            % 若x_low处已为正，则根位于 [t, x_low] 之间
            x_high = x_low;
            x_low = t;      % t处F为负，满足异号条件
        else
            % 否则向右搜索直到函数值变号
            step = 0.1;      % 初始步长
            x_high = x_low + step;
            max_iter = 100;
            for j = 1:max_iter
                f_high = F(x_high);
                if f_high >= 0
                    break;
                end
                x_low = x_high;
                x_high = x_high + step;
                step = step * 1.5;   % 动态增加步长
            end
            if j == max_iter
                error('无法找到包含根的区间，请检查length值或theta');
            end
        end

        % 在区间 [x_low, x_high] 上使用fzero求根（两端函数值异号）
        res(i) = fzero(F, [x_low, x_high]);
    end
end