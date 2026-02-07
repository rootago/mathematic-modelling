function objectiveA = fun(x)
    objectiveA = 0;
    penalty = 0;
    times = [];
    total_smoke = 0;
    for i = 1:2:5
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);

        [t_start,t_end] = getTime(t1,t2,v,theta);
        times = [times;[t_start, t_end]];
        t_smoke = t_end - t_start;
        total_smoke = total_smoke + t_smoke;

        fprintf('屏蔽时间：[%f]\n ', t_smoke);
        fprintf('屏蔽区间：[%f,%f]\n ', t_start, t_end);

        % 约束条件2 - 但要避免越界
        if i <= 3  % 只对前两对检查，避免访问x(7)
            t_prev = x(i);
            t_post = x(i+2);
            if t_post - t_prev < 1
                penalty = penalty + 10;
                fprintf('违反约束2：|%f - %f| < 1\n', t_post, t_prev);
            end
        end

        if t_smoke<0.1
            penalty = penalty+10;
        end
    end

    total_time = mergeTime(times);

    objectiveA = -total_time + penalty;
    fprintf('目标值：[%f] ', -objectiveA);

end  
