% 设定常数
g = 9.8;

% 五组x值
x_values = {
    [0.01, 0.02, 2.38, 0.00, 1.23, 0.00, 83.03, 3.2971];
    [3.72, 6.86, 2.71, 4.47, 19.96, 20.00, 132.66, 4.4240];
    [1.16, 2.45, 3.06, 2.58, 1.54, 8.44, 108.59, 4.1102];
    [8.11, 11.15, 4.15, 10.17, 0.02, 0.25, 111.49, 5.1169];
    [13.81, 0.08, 8.84, 6.93, 19.73, 6.77, 115.25, 2.4212]
};

% 无人机位置
BPos0 = zeros(5,3);
BPos0(1,:) = [17800, 0, 1800];
BPos0(2,:) = [12000, 1400, 1400];
BPos0(3,:) = [6000, -3000, 700];
BPos0(4,:) = [11000, 2000, 1800];
BPos0(5,:) = [13000, -2000, 1300];

% 五次循环计算
for uav_idx = 1:5
    x = x_values{uav_idx};
    fprintf('\n===== 第%d架无人机 =====\n', uav_idx);
    fprintf('无人机位置: [%.0f, %.0f, %.0f]\n', BPos0(uav_idx,:));
    fprintf('参数x: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.4f]\n', x);
    
    % 计算投放点坐标
    drop_pos = [];
    for i = 1:2:5
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        
        DPos = [BPos0(uav_idx,1) + v*t1*cos(theta),...
                BPos0(uav_idx,2) + v*t1*sin(theta),...
                BPos0(uav_idx,3)];
        drop_pos = [drop_pos; DPos];
    end
    
    % 计算起爆点坐标
    explode_pos = [];
    for i = 1:2:5
        t1 = x(i);
        t2 = x(i+1);
        v = x(7);
        theta = x(8);
        
        DPos = [BPos0(uav_idx,1) + v*(t1+t2)*cos(theta),...
                BPos0(uav_idx,2) + v*(t1+t2)*sin(theta),...
                BPos0(uav_idx,3) - 0.5*g*t2^2 - 3*(t1+t2)];
        explode_pos = [explode_pos; DPos];
    end
    
    fprintf('投放点坐标:\n');
    for j = 1:3
        fprintf('  干扰弹%d: [%.2f, %.2f, %.2f]\n', j, drop_pos(j,:));
    end
    
    fprintf('起爆点坐标:\n');
    for j = 1:3
        fprintf('  干扰弹%d: [%.2f, %.2f, %.2f]\n', j, explode_pos(j,:));
    end
end