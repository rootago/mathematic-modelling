clear;clc
clear global global_smoke_intervals curr_b
%% 贪心优化参数
num_b = 5;
global_smoke_intervals = cell(3,1);
for i = 1:3
 global_smoke_intervals{i} = []; % 初始化为空
end
% GA选项
options = optimoptions('ga', ...
        'Display','iter', ...
        'PopulationSize', 200, ...       
        'MaxGenerations', 500, ...        
        'MaxStallGenerations', 100, ...   
        'MutationFcn', {@mutationuniform, 0.5},...  
        'CrossoverFcn', {@crossoverscattered}, ...   
        'SelectionFcn', {@selectiontournament, 4},... 
        'EliteCount', 10, ...   
        'FunctionTolerance', 1e-15, ...  
        'ConstraintTolerance', 1e-6);    

% 单架无人机的参数 
x0_single = [3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 105, pi];      
lb_single = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 70, 0]; 
ub_single = [15.0, 20.0, 15.0, 20.0, 15.0, 20.0, 140, 2*pi];
nvars_single = 8;
%% 全局变量
global curr_b global_smoke_intervals
optimal_solutions = [];
%% 贪心优化循环
for b = 1:num_b
     fprintf('\n优化第 %d 架无人机\n', b);
     % 设置当前无人机索引为全局变量
     curr_b = b;
     % 优化当前无人机
     [x,fval] = ga(@fun, nvars_single, [], [], [], [], lb_single, ub_single, [], options);
     % 优化完成后更新全局变量
     updateGlobalIntervals(x, curr_b);
     % 保存解
     optimal_solutions = [optimal_solutions; x];
     % 输出结果
     fprintf('第%d架无人机最优解：\n', b);
     disp(x);
     fprintf('第%d架无人机目标值：%.4f\n', b, -fval);
end
fprintf('\n最终结果\n');
% 输出最终三枚导弹的累积遮蔽时间区间
for i = 1:3
     fprintf('导弹%d的累积遮蔽时间区间：\n', i);
    if ~isempty(global_smoke_intervals{i})
        total_time_missile = 0;
        for j = 1:size(global_smoke_intervals{i}, 1)
             fprintf('[%.2f, %.2f] ', global_smoke_intervals{i}(j,1), global_smoke_intervals{i}(j,2));
            total_time_missile = total_time_missile + (global_smoke_intervals{i}(j,2) - global_smoke_intervals{i}(j,1));
        end
    fprintf('\n导弹%d总遮蔽时间：%.4f\n', i, total_time_missile);
    else
        fprintf('无遮蔽区间\n');
    end
end