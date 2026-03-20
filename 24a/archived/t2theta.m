% 龙头轨迹，由时间得极坐标辐角
function res = t2theta(t)
    % t = t(:); % 确保 t 是列向量
    % t = t';
    load('constants.mat');
    a = PITCH / (2 * PI); % 系数 55/2pi
    v = VELOCITY; % 线速度 100
    INIT_THETA = INIT_POS * 2 * PI;
    F = @(x) x .* sqrt(1 + x.^2) + asinh(x);
    C = F(INIT_THETA);
    target = 2 * v / a * t;
    res = zeros(size(t));
    for i = 1:numel(t)
        res(i) = fzero(@(x) F(x) - C + target(i), INIT_THETA);
    end
end