clear; clc;

%% 完全清除所有全局变量
clear global;

%% 贪心优化参数
num_b = 5;

% 重新初始化全局变量
global global_smoke_intervals curr_b;
global_smoke_intervals = cell(3,1);
for i = 1:3
    global_smoke_intervals{i} = []; % 确保为空矩阵
end

% PSO选项 - 优化参数以增强搜索能力
options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'SwarmSize', 200, ...              % 增大粒子群规模
    'MaxIterations', 500, ...          % 增加最大迭代次数  
    'MaxStallIterations', 100, ...     % 增加停滞迭代次数
    'FunctionTolerance', 1e-15, ...    % 更严格的函数容忍度
    'InertiaRange', [0.2, 0.9], ...    % 调整惯性权重范围
    'SelfAdjustmentWeight', 2.0, ...   % 个体学习因子
    'SocialAdjustmentWeight', 2.0);    % 群体学习因子

% 单架无人机的参数 - 每架无人机使用不同的优化初值
x0_all = [
    4.76, 7.82, 7.82, 11.69, 11.69, 13.50, 140, 3.140732038;  % FY1初值
    15.57, 17.85, 17.85, 21.07, 21.07, 21.08, 140, 5.594006064; % FY2初值
    22.54, 27.65, 28.53, 32.80, 32.80, 36.64, 140, 1.369491639; % FY3初值
    48.80, 66.00, 66.19, 66.20, 66.37, 66.40, 140, 3.376021901; % FY4初值
    14.57, 17.43, 17.43, 21.60, 21.60, 24.83, 140, 1.68133892   % FY5初值
];

% 根据初值计算个性化边界 - 以初值为中心±5浮动
lb_all = max(0, x0_all - 5);  % 下界：初值-5，但不小于0
ub_all = x0_all + 5;          % 上界：初值+5

% 特殊处理速度和角度的边界
lb_all(:, 7) = max(70, lb_all(:, 7));    % 速度下界不低于70
ub_all(:, 7) = min(140, ub_all(:, 7));   % 速度上界不超过140
lb_all(:, 8) = max(0, lb_all(:, 8));     % 角度下界不小于0
ub_all(:, 8) = min(2*pi, ub_all(:, 8));  % 角度上界不超过2π

nvars_single = 8;
optimal_solutions = [];

%% 验证初始状态
fprintf('开始优化前的全局变量状态：\n');
for i = 1:3
    fprintf('导弹%d初始区间数量：%d\n', i, size(global_smoke_intervals{i}, 1));
end

%% 贪心优化循环
for b = 1:num_b
    fprintf('\n=== 开始优化第 %d 架无人机 ===\n', b);
    
    % 显示当前全局状态
    fprintf('优化前全局区间状态：\n');
    for i = 1:3
        if ~isempty(global_smoke_intervals{i})
            fprintf('导弹%d现有区间：', i);
            for j = 1:size(global_smoke_intervals{i}, 1)
                fprintf('[%.2f, %.2f] ', global_smoke_intervals{i}(j,1), global_smoke_intervals{i}(j,2));
            end
            fprintf('\n');
        else
            fprintf('导弹%d现有区间：无\n', i);
        end
    end
    
    % 设置当前无人机索引
    curr_b = b;
    
    % 备份当前全局状态（防止PSO过程中的多次更新）
    global_backup = global_smoke_intervals;
    
    % 根据当前无人机选择对应的初值和边界
    x0_single = x0_all(b, :);     % 每架无人机使用自己的初值
    lb_single = lb_all(b, :);     % 每架无人机使用自己的下界
    ub_single = ub_all(b, :);     % 每架无人机使用自己的上界
    
    fprintf('使用第%d架无人机的专用初值: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.1f, %.3f]\n', b, x0_single);
    fprintf('对应的搜索边界: [%.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.0f-%.0f, %.2f-%.2f]\n', ...
        lb_single(1), ub_single(1), lb_single(2), ub_single(2), lb_single(3), ub_single(3), ...
        lb_single(4), ub_single(4), lb_single(5), ub_single(5), lb_single(6), ub_single(6), ...
        lb_single(7), ub_single(7), lb_single(8), ub_single(8));
    
    % 创建包含好初值的初始群体
    swarm_size = 200;
    initial_swarm = repmat(x0_single, swarm_size, 1);
    % 在初值基础上添加随机扰动
    for i = 1:swarm_size
        for j = 1:nvars_single
            % 在边界内随机扰动 ±20%
            range = ub_single(j) - lb_single(j);
            perturbation = (rand - 0.5) * 0.4 * range;
            initial_swarm(i, j) = max(lb_single(j), min(ub_single(j), x0_single(j) + perturbation));
        end
    end
    
    % 更新PSO选项以使用初始群体
    options_with_init = optimoptions(options, 'InitialSwarmMatrix', initial_swarm);

    % 优化当前无人机
    [x, fval] = particleswarm(@fun, nvars_single, lb_single, ub_single, options_with_init);
    
    % 恢复全局状态，然后只更新一次
    global_smoke_intervals = global_backup;

    % 手动更新全局变量（只更新一次）
    updateGlobalIntervals(x, curr_b);

    % 显示当前无人机的优化结果
    fprintf('\n第%d架无人机最优解：\n', b);
    disp(x);
    fprintf('第%d架无人机目标值：%.4f\n', b, -fval);
    
    % 手动调用一次目标函数来获取详细信息
    fprintf('\n验证第%d架无人机的遮蔽效果：\n', b);
    test_fval = fun(x);
    fprintf('验证目标值：%.4f\n', -test_fval);
    
    % 保存解
    optimal_solutions = [optimal_solutions; x];
    
    fprintf('=== 第 %d 架无人机优化完成 ===\n', b);
end

%% 最终结果统计
fprintf('\n=================== 最终结果 ===================\n');
total_coverage_time = 0;

for i = 1:3
    fprintf('\n导弹%d的累积遮蔽时间区间：\n', i);
    if ~isempty(global_smoke_intervals{i})
        total_time_missile = 0;
        fprintf('区间详情：');
        for j = 1:size(global_smoke_intervals{i}, 1)
            interval_time = global_smoke_intervals{i}(j,2) - global_smoke_intervals{i}(j,1);
            fprintf('[%.2f, %.2f](%.2f秒) ', ...
                global_smoke_intervals{i}(j,1), global_smoke_intervals{i}(j,2), interval_time);
            total_time_missile = total_time_missile + interval_time;
        end
        fprintf('\n导弹%d总遮蔽时间：%.4f秒\n', i, total_time_missile);
        total_coverage_time = total_coverage_time + total_time_missile;
    else
        fprintf('无遮蔽区间\n');
    end
end

fprintf('\n所有导弹总遮蔽时间：%.4f秒\n', total_coverage_time);
fprintf('平均每枚导弹遮蔽时间：%.4f秒\n', total_coverage_time/3);

%% 显示所有无人机的解
fprintf('\n=================== 所有无人机解 ===================\n');
for b = 1:num_b
    fprintf('无人机%d: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.4f]\n', ...
        b, optimal_solutions(b,:));
end