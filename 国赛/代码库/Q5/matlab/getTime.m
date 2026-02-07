function [t_start, t_end] = getTime(t1, t2, v, theta,APos0,BPos0)
    % 参数
    AVel = 300;
    alpha1 = atan(APos0(3)/sqrt(APos0(1)^2 + APos0(2)^2));
    ADir = [-cos(alpha1), 0, -sin(alpha1)];
    g = 9.8;
    
    % 遍历所有点
    points = generate_points();
    times = [];
for i = 1:size(points,1)
    OPos = points(i,:);
    % fprintf('采点位置[%d]：[%f, %f, %f] ', i, OPos);
    % 仿真模拟
    t_step = 0.1;
    t = 0;
    flag1 = 0;
    flag2 = 0;
    current_t_start = 0;
    current_t_end = 0;
    while t <= 20
        %%
        % if t >= 4.7 && t <= 4.9
        %     t_step = 1e-4;
        % else
        %     t_step = 0.1;
        % end
        %%
        % 计算导弹位置
        APos = APos0 + (t+t1+t2)*ADir*AVel;
        % 计算烟幕弹位置
        DPos = [BPos0(1) + v*(t1+t2)*cos(theta),...
                BPos0(2) + v*(t1+t2)*sin(theta),...
                BPos0(3) - 0.5*g*t2^2 - 3*t];

        % 计算距离
        OA = APos - OPos;
        OD = DPos - OPos;
        distance1 = norm(cross(OA,OD))/norm(OA);
        distance2 = norm(APos - DPos);
        
        % 添加调试信息
        % if t == 0
        %     fprintf('t=0: APos=[%.1f,%.1f,%.1f], DPos=[%.1f,%.1f,%.1f], d1=%.1f, d2=%.1f\n', ...
        %             APos, DPos, distance1, distance2);
        % end
        if flag1 ==0 && distance1 <= 10
           flag1 = 1;
           current_t_start = t+t1+t2;
        end
        if flag1 == 1 && distance1 > 10
           current_t_end = t+t1+t2;
           break;
        end
        
        if distance2 <= 10  % 改回题目要求的10m遮蔽半径
           flag2 = 1;
        end
        if flag2 == 1 && distance2 > 10  % 改回题目要求的10m遮蔽半径
            current_t_end = t+t1+t2;
            break;
        end

        % 手动更新时间
        t = t + t_step;  
    end
    if current_t_start == 0 && current_t_end == 0
        current_time = [-1, -1];  % 用负数表示无效
    else
        current_time = [current_t_start, current_t_end];
    end
    % fprintf('current_time：[%f, %f]\n ',current_time);
    times = [times;current_time];
    
end
    % 过滤无效区间
    valid_times = times(times(:,1) > 0, :);
    if isempty(valid_times)
        t_start = 0; 
        t_end = 0;
    else
        t_start = max(valid_times(:,1));
        t_end = min(valid_times(:,2));
    end
end