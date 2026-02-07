function objectiveA = fun(x)
    global curr_b global_smoke_intervals
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

    for i = 1:2:5 %三个烟雾弹
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        
        % 每个烟雾弹对所有三枚导弹都计算遮蔽区间
        for missile_idx = 1:3
            [t_start,t_end] = getTime(t1,t2,v,theta,APos0(missile_idx,:),BPos0(curr_b,:));
            
            if t_start < t_end  % 确保时间区间有效
                current_intervals{missile_idx} = [current_intervals{missile_idx}; [t_start, t_end]];
            end
        end

        % 约束条件
        if i <= 3  
            t_prev = x(i);
            t_post = x(i+2);
            if abs(t_post - t_prev) < 1
                penalty = penalty + 10000;
                fprintf('违反约束：|%f - %f| < 1 ', t_post, t_prev);
            end
        end
        
    end

    total_time = 0;
    for missile_idx = 1:3
        if ~isempty(current_intervals{missile_idx})
            if isempty(global_smoke_intervals{missile_idx})
                % 如果全局区间为空，只计算当前区间的时间
                merged_intervals = mergeTime(current_intervals{missile_idx});
            else
                % 合并全局区间和当前区间
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


     % 打印每个烟雾弹对所有导弹的遮蔽情况
    smoke_bomb_index = 1;
    for i = 1:2:5 % 三个烟雾弹
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        
        fprintf("***************************************\n");
        fprintf("第%d架无人机的第%d个烟雾弹 (t1=%.2f, t2=%.2f):\n", curr_b, smoke_bomb_index, t1, t2);
        
        % 该烟雾弹对所有三枚导弹的遮蔽情况
        for missile_idx = 1:3
            [t_start,t_end] = getTime(t1,t2,v,theta,APos0(missile_idx,:),BPos0(curr_b,:));
            
            fprintf("  对第%d枚导弹: ", missile_idx);
            if t_start < t_end 
                fprintf("[%.4f, %.4f] (持续%.4f秒)\n", t_start, t_end, t_end-t_start);
            else
                fprintf("无遮蔽 (t_start=%.4f >= t_end=%.4f)\n", t_start, t_end);
            end
        end
        
        smoke_bomb_index = smoke_bomb_index + 1;
    end
end  
