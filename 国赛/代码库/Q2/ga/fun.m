function objectiveA = fun(x)
    t1 = x(1);
    t2 = x(2);
    v = x(3);
    theta = x(4);
   

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

    objectiveA = 0;
    penalty = 0;
    t_smoke = 0;

    [t_start,t_end] = getTime(t1,t2,v,theta,APos0(1,:),BPos0(1,:));
    t_smoke = t_end - t_start;

    fprintf('屏蔽时间：[%f]\n ', t_smoke);
    fprintf('屏蔽区间：[%f,%f]\n ', t_start, t_end);
    
    objectiveA = -t_smoke + penalty;

    fprintf('目标值：[%f] ', -objectiveA);
end  
      