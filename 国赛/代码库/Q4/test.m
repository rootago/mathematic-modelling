% 分析零值现象
clear;clc

% 当前最优解
x_optimal = [2.5423, 11.8823, 97.1952, 2.0274, 0.0463, 1.0638, 72.4178, 0.1792, 0.1852, 0.3456, 98.0616, 3.1378];
current_value = -fun(x_optimal);

fprintf('=== 零值现象分析 ===\n');
fprintf('当前最优值: %.6f\n', current_value);
fprintf('当前解中的小值: ');
small_indices = find(x_optimal < 1);
fprintf('x(%d)=%.4f ', [small_indices; x_optimal(small_indices)]);
fprintf('\n\n');

%% 测试1：将小值增大看效果
fprintf('=== 测试1：将小值适当增大 ===\n');

% 边界
lb = [0, 0, 70, 0, 0, 0, 70, 0, 0, 0, 70, 0];
ub = [12, 12, 140, pi, 12, 12, 140, pi, 12, 12, 140, pi];

% 创建几个测试点
test_cases = {
    '将x5从0.0463改为1', [2.5423, 11.8823, 97.1952, 2.0274, 1.0000, 1.0638, 72.4178, 0.1792, 0.1852, 0.3456, 98.0616, 3.1378];
    '将x8从0.1792改为1', [2.5423, 11.8823, 97.1952, 2.0274, 0.0463, 1.0638, 72.4178, 1.0000, 0.1852, 0.3456, 98.0616, 3.1378];
    '将x9从0.1852改为1', [2.5423, 11.8823, 97.1952, 2.0274, 0.0463, 1.0638, 72.4178, 0.1792, 1.0000, 0.3456, 98.0616, 3.1378];
    '将x10从0.3456改为1', [2.5423, 11.8823, 97.1952, 2.0274, 0.0463, 1.0638, 72.4178, 0.1792, 0.1852, 1.0000, 98.0616, 3.1378];
    '将所有小值都改为1', [2.5423, 11.8823, 97.1952, 2.0274, 1.0000, 1.0638, 72.4178, 1.0000, 1.0000, 1.0000, 98.0616, 3.1378];
};

for i = 1:size(test_cases, 1)
    test_x = test_cases{i, 2};
    test_value = -fun(test_x);
    improvement = test_value - current_value;
    
    fprintf('%s:\n', test_cases{i, 1});
    fprintf('  目标值: %.6f, 改进: %+.6f\n', test_value, improvement);
end

%% 测试2：梯度方向测试
fprintf('\n=== 测试2：简单梯度分析 ===\n');

% 对小值变量进行微小扰动，看目标函数变化
delta = 0.01;
for idx = small_indices
    x_plus = x_optimal;
    x_plus(idx) = x_plus(idx) + delta;
    
    % 检查边界
    if x_plus(idx) <= ub(idx)
        value_plus = -fun(x_plus);
        gradient_approx = (value_plus - current_value) / delta;
        
        fprintf('变量x(%d): 当前值=%.4f, 梯度≈%.4f', idx, x_optimal(idx), gradient_approx);
        if gradient_approx > 0
            fprintf(' (增大会改进!)');
        elseif gradient_approx < 0
            fprintf(' (增大会变差)');
        else
            fprintf(' (无影响)');
        end
        fprintf('\n');
    end
end

%% 测试3：尝试更激进的改变
fprintf('\n=== 测试3：激进测试 ===\n');

% 将所有小于1的值都设为较大值
x_aggressive = x_optimal;
for idx = small_indices
    if idx ~= 3 && idx ~= 7 && idx ~= 11  % 跳过角度约束较强的变量
        x_aggressive(idx) = min(ub(idx), 3.0);  % 设为3或上界
    end
end

fprintf('激进方案 (小值都改为3或上界):\n');
fprintf('原值: ');
fprintf('%.3f ', x_optimal);
fprintf('\n新值: ');
fprintf('%.3f ', x_aggressive);
fprintf('\n');

value_aggressive = -fun(x_aggressive);
fprintf('激进方案目标值: %.6f\n', value_aggressive);
fprintf('相比原解改进: %+.6f\n', value_aggressive - current_value);

if value_aggressive > current_value
    fprintf('🎉 激进方案确实更优！说明原解确实有问题\n');
    
    % 基于这个更好的点再次优化
    fprintf('\n=== 基于激进方案继续优化 ===\n');
    
    options = optimoptions('ga', ...
        'Display', 'iter', ...
        'PopulationSize', 300, ...
        'MaxGenerations', 1000, ...
        'MaxStallGenerations', 200, ...
        'FunctionTolerance', 1e-12, ...
        'InitialPopulationMatrix', repmat(x_aggressive, 1, 1)', ...
        'UseParallel', true);
    
    [x_new, fval_new] = ga(@fun, 12, [], [], [], [], lb, ub, [], options);
    
    fprintf('\n进一步优化结果:\n');
    fprintf('目标值: %.8f\n', -fval_new);
    fprintf('相比原解改进: %+.8f\n', -fval_new - current_value);
    
    disp('新的最优解：');
    disp(x_new);
else
    fprintf('😮 激进方案反而更差，说明确实存在复杂约束\n');
end