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
    for i = 1:2:5 % 1 3 5
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        % 导弹索引
        j = ceil(i/2);% 1 2 3
        [t_start,t_end] = getTime(t1,t2,v,theta,APos0(j,:),BPos0(1,:));
        times = [times;[t_start, t_end]];
        t_smoke = t_end - t_start;
        total_smoke = total_smoke + t_smoke;

        fprintf('屏蔽时间：[%f]\n ', t_smoke);
        fprintf('屏蔽区间：[%f,%f]\n ', t_start, t_end);

        % 约束条件2
        if i <= 3  % 只对前两对检查，避免访问x(7)
            t_prev = x(i);
            t_post = x(i+2);
            if abs(t_post - t_prev) < 1
                penalty = penalty + 10000;
                fprintf('违反约束2：|%f - %f| < 1\n', t_post, t_prev);
            end
        end
        
    end

    total_time = mergeTime(times);

    objectiveA = -total_time + penalty;
    fprintf('目标值：[%f] ', -objectiveA);

end  
