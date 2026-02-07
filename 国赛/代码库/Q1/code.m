clear;clc
% 圆柱参数
center = [0,200,0];   % 底面圆心
R = 7;                % 半径
H = 10; 

% 采样精度
n_r = 5;       % 半径方向点数
n_theta = 37;  % 角度方向点数
n_h = 11;      % 高度方向点数

points = [];
% 侧面采样
thetas = linspace(0,2*pi,n_theta);
hs = linspace(0,H,n_h);
for theta = thetas
    for h = hs
        x = center(1) + R*cos(theta);
        y = center(2) + R*sin(theta);
        z = center(3) + h;
        points(end+1,:) = [x,y,z];
    end
end
% disp(length(points)); 396
% 上底面采样 (z = z0 + H)
rs = linspace(0,R,n_r);
for r = rs
    for theta = thetas
        x = center(1) + r*cos(theta);
        y = center(2) + r*sin(theta);
        z = center(3) + H;
        points(end+1,:) = [x,y,z];
    end
end

% uav 无人机 missile 导弹 真目标位置
uavPos = [17800, 0, 1800];
missilePos = [20000, 0, 2000];

uavVel = 120;
missileVel = 300;

uavLen = norm(uavPos);
missileLen = norm(missilePos);

uavDir = [-1,0,0];
missileDir = -missilePos/missileLen;

% 烟幕干扰弹投放
t1 = 1.5;
uavPos = uavPos + uavDir*t1*uavVel;
missilePos = missilePos + missileDir*t1*missileVel;

smokeDecoyPos = uavPos;
smokeDecoyVel = uavDir*uavVel; 

fprintf('1.5s 无人机位置：[%f, %f, %f]\n', uavPos);
fprintf('1.5s 导弹位置：[%f, %f, %f]\n', missilePos);
fprintf('1.5s 烟幕干扰弹位置：[%f, %f, %f]\n', smokeDecoyPos);

% 烟幕干扰弹起爆
t2 = 5.1;
dt = t2-t1;
uavPos = uavPos + uavDir*dt*uavVel;
missilePos = missilePos + missileDir*dt*missileVel;
g = 9.8;
smokeDecoyPos = smokeDecoyPos + smokeDecoyVel*dt + 0.5*[0, 0, -g]*dt^2;


fprintf('5.1s 无人机位置：[%f, %f, %f]\n', uavPos);
fprintf('5.1s 导弹位置：[%f, %f, %f]\n', missilePos);
fprintf('5.1s 烟幕干扰弹位置：[%f, %f, %f]\n', smokeDecoyPos);

smokeDecoyVel = [0, 0, -3];

% 保留5.1s的位置
missilePos0 = missilePos;
smokeDecoyPos0 = smokeDecoyPos;
% 计算遮蔽的时间
times = [];
for i = 1:size(points,1)
    targetPos = points(i,:);
    fprintf('采点位置[%d]：[%f, %f, %f]\n', i, targetPos);
    
    % 重新初始化
    missilePos = missilePos0;
    smokeDecoyPos = smokeDecoyPos0;
    is_start = false;
    is_end = false;
    t_start = 0;
    t_end = 0;
    
    t = 5.1;  % 初始时间
    while t <= 20
        % fprintf('时间：[%f] ', t);
        
        % 动态调整步长
        if t >= 7.5 && t <= 10.0
            d_step = 0.000001;
        else
            d_step = 0.01;
        end
        
        if t > 5.1
            % 更新导弹的位置
            missilePos = missilePos + missileVel*missileDir*d_step;
            % 更新烟幕干扰弹的位置
            smokeDecoyPos = smokeDecoyPos + smokeDecoyVel*d_step;
        end
        
        % 距离计算和判断逻辑
        lineDir = targetPos - missilePos;
        distance1 = norm(cross(smokeDecoyPos - missilePos, lineDir)) / norm(lineDir);
        distance2 = norm(smokeDecoyPos - missilePos);
        
        if ~is_start && distance1 <= 10
            t_start = t;
            is_start = true;
        elseif is_start && ~is_end && distance2 < 10
            is_end = true;
        elseif is_start && is_end && distance2 > 10
            t_end = t;
            break;
        end

        t = t + d_step;  % 手动更新时间
    end
    
    times(end+1,:) = [t_start, t_end];
    disp([t_start,t_end]);
end

% 计算交集
[t_start_final, idx_start] = max(times(:,1));
[t_end_final, idx_end] = min(times(:,2));

fprintf('遮蔽区间交集: [%f, %f]\n', t_start_final, t_end_final);
fprintf('遮蔽时间: [%f]\n', t_end_final- t_start_final);

fprintf('采点位置[%d]：[%f, %f, %f]\n', idx_start, points(idx_start));
fprintf('采点位置[%d]：[%f, %f, %f]\n', idx_end, points(idx_end));