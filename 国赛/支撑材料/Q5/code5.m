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
    'SwarmSize', 200, ...
    'MaxIterations', 500, ...
    'MaxStallIterations', 100, ...
    'FunctionTolerance', 1e-15, ...
    'InertiaRange', [0.2, 0.3], ...
    'SelfAdjustmentWeight', 2.0, ...
    'SocialAdjustmentWeight', 2.0);

% 单架无人机的参数
x0_single = [3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 105, pi]; 
lb_single = [0, 0, 0, 0, 0,0 70, 0];  
ub_single = [20, 20, 20, 20, 20, 20, 140, 2*pi]; 

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
    
    % 备份当前全局状态
    global_backup = global_smoke_intervals;
    
    % 创建初始群体
    swarm_size = 200;
    initial_swarm = repmat(x0_single, swarm_size, 1);
    for i = 1:swarm_size
        for j = 1:nvars_single
            range = ub_single(j) - lb_single(j);
            perturbation = (rand - 0.5) * 0.4 * range;
            initial_swarm(i, j) = max(lb_single(j), min(ub_single(j), x0_single(j) + perturbation));
        end
    end
    
    % 使用初始群体
    options_with_init = optimoptions(options, 'InitialSwarmMatrix', initial_swarm);

    % 优化当前无人机
    [x, fval] = particleswarm(@fun, nvars_single, lb_single, ub_single, options_with_init);
    
    % 恢复全局状态
    global_smoke_intervals = global_backup;

    % 更新全局变量
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
    curr_b = b;
    fun(optimal_solutions(b,:));
end

