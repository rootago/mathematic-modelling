function objectiveA = fun(x)
    global curr_b global_smoke_intervals
    fprintf('DEBUG: fun被调用，当前global_smoke_intervals{1}的大小: %d\n', size(global_smoke_intervals{1}, 1));
    objectiveA = 0;
    penalty = 0;
    times = [];
    % 导弹
    APos0 = zeros(3,3); 
    APos0(1,:) = [20000, 0, 2000];
    APos0(2,:) = [19000, 600, 2100];
    APos0(3,:) = [18000, -600, 1900];
    % 无人机
    BPos0 = zeros(5,3);
    BPos0(1,:) = [17800, 0, 1800];
    BPos0(2,:) = [12000, 1400, 1400];
    BPos0(3,:) = [6000, -3000, 700];
    BPos0(4,:) = [11000, 2000, 1800];
    BPos0(5,:) = [13000, -2000, 1300];

    current_intervals = cell(3,1);  % 当前无人机对三枚导弹的遮蔽区间
    for i = 1:3
        current_intervals{i} = [];
    end
    
    % 分别对三个导弹计算时间
    for j = 1:3 % 对每枚导弹
        fprintf('\n第%d枚导弹的遮蔽区间\n', j);
        missile_intervals = []; % 当前导弹的所有烟雾弹遮蔽区间
        
        % 对当前导弹，计算三个烟雾弹的遮蔽时长
        for i = 1:2:5 % 1, 3, 5 对应三个烟雾弹参数对
            smoke_idx = (i+1)/2; % 烟雾弹索引：1, 2, 3
            t1 = x(i);
            t2 = x(i+1);
            v = x(7);
            theta = x(8);
            
            fprintf('第%d架无人机，第%d枚导弹，第%d个烟雾弹：\n', curr_b, j, smoke_idx);
            [t_start, t_end] = getTime(t1, t2, v, theta, APos0(j,:), BPos0(curr_b,:));
            fprintf('时间区间：[%.4f, %.4f]\n', t_start, t_end);
            
            % 如果时间区间有效，添加到当前导弹的区间列表
            if t_start < t_end && t_start > 0
                missile_intervals = [missile_intervals; [t_start, t_end]];
            end
        end
        
        % 合并当前导弹的所有烟雾弹区间（求并集）
        if ~isempty(missile_intervals)
            merged_missile_intervals = mergeTime(missile_intervals);
            current_intervals{j} = merged_missile_intervals;
            fprintf('导弹%d合并后区间：', j);
            for k = 1:size(merged_missile_intervals, 1)
                fprintf('[%.4f, %.4f] ', merged_missile_intervals(k,1), merged_missile_intervals(k,2));
            end
            fprintf('\n');
        end
    end
    
    % 约束条件2：检查相邻烟雾弹投放时间间隔
    for i = 1:2:3  % 检查第1-2和第2-3烟雾弹的时间间隔
        t_curr = x(i);      % 当前烟雾弹的t1
        t_next = x(i+2);    % 下一个烟雾弹的t1
        if abs(t_next - t_curr) < 1
            penalty = penalty + 10000;
            fprintf('违反约束2：第%d和第%d烟雾弹间隔 |%.4f - %.4f| = %.4f < 1\n', ...
                (i+1)/2, (i+3)/2, t_curr, t_next, abs(t_next - t_curr));
        end
    end

    total_time = 0;
    for missile_idx = 1:3
        % 关键修改：不更新全局变量，只计算合并后的时间
        if ~isempty(current_intervals{missile_idx})
            if isempty(global_smoke_intervals{missile_idx})
                % 如果全局区间为空，直接使用当前区间（已经合并过了）
                merged_intervals = current_intervals{missile_idx};
            else
                % 合并全局区间和当前区间，但不保存到全局变量
                combined_intervals = [global_smoke_intervals{missile_idx}; current_intervals{missile_idx}];
                merged_intervals = mergeTime(combined_intervals);
            end
            
            % 计算合并后区间的总时间
            for k = 1:size(merged_intervals, 1)
                total_time = total_time + (merged_intervals(k,2) - merged_intervals(k,1));
            end
        else
            % 如果当前无人机对该导弹没有贡献，只计算全局区间的时间
            if ~isempty(global_smoke_intervals{missile_idx})
                for k = 1:size(global_smoke_intervals{missile_idx}, 1)
                    total_time = total_time + (global_smoke_intervals{missile_idx}(k,2) - global_smoke_intervals{missile_idx}(k,1));
                end
            end
        end
    end
    
     objectiveA = -total_time + penalty;
     fprintf('目标值：[%f] ', -objectiveA);

end  
