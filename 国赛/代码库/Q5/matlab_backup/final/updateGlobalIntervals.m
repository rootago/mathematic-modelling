function updateGlobalIntervals(x, curr_b)
    global global_smoke_intervals
    
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
    
    current_intervals = cell(3,1);
    for i = 1:3
        current_intervals{i} = [];
    end
    
    for i = 1:2:5 % 1 3 5
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        j = ceil(i/2);
        [t_start,t_end] = getTime(t1,t2,v,theta,APos0(j,:),BPos0(curr_b,:));
        if t_start < t_end
            current_intervals{j} = [current_intervals{j}; [t_start, t_end]];
        end
    end
    
    % 更新全局变量
    fprintf('\n第%d架无人机优化完成，更新全局区间：\n', curr_b);
    for missile_idx = 1:3
        if ~isempty(current_intervals{missile_idx})
            if isempty(global_smoke_intervals{missile_idx})
                global_smoke_intervals{missile_idx} = mergeTime(current_intervals{missile_idx});
            else
                combined_intervals = [global_smoke_intervals{missile_idx}; current_intervals{missile_idx}];
                global_smoke_intervals{missile_idx} = mergeTime(combined_intervals);
            end
            fprintf('导弹%d更新后的全局区间大小: %d\n', missile_idx, size(global_smoke_intervals{missile_idx}, 1));
        end
    end
end