function objectiveA = fun(x)
    objectiveA = 0;
    penalty = 0;
    times = [];
    total_smoke = 0;

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

    for i = 1:4:9
        % 计算无人机索引
        j = ceil(i/4);
        t1 = x(i);
        t2 = x(i+1);
        v = x(i+2);
        theta = x(i+3);

        [t_start,t_end] = getTime(t1,t2,v,theta,APos0(1,:),BPos0(j,:));
        times = [times;[t_start, t_end]];
        t_smoke = t_end - t_start;
        % total_smoke = total_smoke + t_smoke;

        fprintf('屏蔽时间：[%f]\n ', t_smoke);
        fprintf('屏蔽区间：[%f,%f]\n ', t_start, t_end);
        
        % % 约束条件1
        % if t1 >= t2 
        %     penalty = penalty + 10000;
        %     fprintf('违反约束1：t1(%f) > t2(%f)\n', t1, t2);
        % end
        
          % if t_smoke < 0.1
          %     penalty = penalty+100;
          % end

    end

    total_time = mergeTime(times);
    fprintf('总时间：[%f] ', total_time);
    objectiveA = -total_time + penalty;
    fprintf('目标值：[%f] ', -objectiveA);
    fprintf('惩罚值：[%f] ', penalty);

end  
