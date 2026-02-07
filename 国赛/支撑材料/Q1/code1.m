clear;clc
% 圆柱采样参数
center = [0,200,0];
R = 7;
H = 10; 
n_r = 5;
n_theta = 73;
n_h = 11;

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

% 上底面采样
rs = linspace(0,R,n_r);
for r = rs
    for theta = thetas
        x = center(1) + r*cos(theta);
        y = center(2) + r*sin(theta);
        z = center(3) + H;
        points(end+1,:) = [x,y,z];
    end
end

% 初始状态参数
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
plot_drawn = false;  % 添加标志变量，确保只画一次图
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
        
        if is_start && is_end
            % 只画一次图，第一次满足条件时
            if ~plot_drawn  
                % 绘制遮蔽示意图
                figure('Position', [100, 100, 1400, 600], 'Renderer', 'painters');
                
                % 全景图
                subplot(1, 2, 1);
                hold on;
                
                scatter3(missilePos(1), missilePos(2)*10, missilePos(3), 200, 'r', 'filled', 'o', 'DisplayName', '导弹');
                scatter3(smokeDecoyPos(1), smokeDecoyPos(2)*10, smokeDecoyPos(3), 200, 'b', 'filled', 's', 'DisplayName', '烟幕弹中心');
                scatter3(targetPos(1), targetPos(2), targetPos(3), 200, 'g', 'filled', '^', 'DisplayName', '目标点');
                
                % 观测线
                line_vec = targetPos - missilePos;
                line_length = norm(line_vec);
                line_dir = line_vec / line_length;
                extend_length = 2000;
                line_start = missilePos - line_dir * extend_length;
                line_end = missilePos + line_dir * (line_length + extend_length);
                plot3([line_start(1), line_end(1)], [line_start(2), line_end(2)], [line_start(3), line_end(3)], 'k--', 'LineWidth', 2, 'DisplayName', '观测线');
                
                xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
                title('全景图','FontSize',14);
                grid on; view(-37.5, 30);
                legend('导弹', '烟幕弹中心', '目标点', '观测线', 'Location', 'best');
                
                xlim([0, 25000]);
                ylim([0, 2500]);
                zlim([0, 2500]);
                
                set(gca, 'XTick', 0:5000:25000);
                set(gca, 'YTick', 0:500:2500);
                set(gca, 'ZTick', 0:500:2500);
                
                
                % 烟幕区域放大图
                subplot(1, 2, 2);
                hold on;
                
                scatter3(missilePos(1), missilePos(2)*10, missilePos(3), 200, 'r', 'filled', 'o', 'DisplayName', '导弹');
                scatter3(smokeDecoyPos(1), smokeDecoyPos(2)*10, smokeDecoyPos(3), 200, 'b', 'filled', 's', 'DisplayName', '烟幕弹中心');
                
                % 烟幕云团 
                [X, Y, Z] = sphere(40); 
                X = X * 10 + smokeDecoyPos(1);
                Y = Y * 100 + smokeDecoyPos(2)*10;
                Z = Z * 10 + smokeDecoyPos(3);
                h_surf = surf(X, Y, Z, 'FaceColor', 'cyan', 'FaceAlpha', 0.7, 'EdgeColor', 'blue', 'EdgeAlpha', 0.5, 'DisplayName', '烟幕云团');
                
                theta = linspace(0, 2*pi, 50);

                % 烟幕边界圆周
                x_h = smokeDecoyPos(1) + 10 * cos(theta);
                y_h = smokeDecoyPos(2)*10 + 100 * sin(theta);
                z_h = smokeDecoyPos(3) * ones(size(theta));
                plot3(x_h, y_h, z_h, 'c-', 'LineWidth', 2, 'DisplayName', '烟幕边界');
                
                phi = linspace(0, 2*pi, 50);
                y_v = smokeDecoyPos(2)*10 + 100 * cos(phi);
                z_v = smokeDecoyPos(3) + 10 * sin(phi);
                x_v = smokeDecoyPos(1) * ones(size(phi));
                plot3(x_v, y_v, z_v, 'm-', 'LineWidth', 2, 'HandleVisibility', 'off');
                
                scatter3(targetPos(1), targetPos(2), targetPos(3), 200, 'g', 'filled', '^', 'DisplayName', '目标点');
                
                % 观测线
                line_vec = targetPos - missilePos;
                line_length = norm(line_vec);
                line_dir = line_vec / line_length;

                extend_length = 100;  
                line_start = missilePos - line_dir * extend_length;
                line_end = targetPos + line_dir * extend_length;
                plot3([line_start(1), line_end(1)], [line_start(2)*10, line_end(2)], [line_start(3), line_end(3)], 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
                
                % 遮蔽示意线
                lineDir = targetPos - missilePos;
                proj_vec = dot(smokeDecoyPos - missilePos, lineDir) / dot(lineDir, lineDir) * lineDir;
                proj_point = missilePos + proj_vec;
                plot3([smokeDecoyPos(1), proj_point(1)], [smokeDecoyPos(2)*10, proj_point(2)], [smokeDecoyPos(3), proj_point(3)], 'r:', 'LineWidth', 3, 'HandleVisibility', 'off');
                
                xlim([smokeDecoyPos(1)-25, smokeDecoyPos(1)+25]);
                ylim([smokeDecoyPos(2)*10-250, smokeDecoyPos(2)*10+250]);  
                zlim([smokeDecoyPos(3)-25, smokeDecoyPos(3)+25]);
                
                xlabel('X (m)', 'FontSize', 12);
                ylabel('Y (m)', 'FontSize', 12);
                zlabel('Z (m)', 'FontSize', 12);
                title(sprintf('烟幕区域放大图\n时间%.3fs, 遮蔽距离%.2fm', t, distance1), 'FontSize', 14);
                legend('show', 'Location', 'best', 'FontSize', 10);
                grid on;
                view(45, 30);
                
                lighting gouraud;
                light('Position', [1 1 1]);
                
                drawnow;
                pause(0.1);
                
                % 保存图片
                try
                    print(gcf, '烟幕遮蔽示意图', '-dpng', '-r300');
                    fprintf('图片已保存为: 烟幕遮蔽示意图.png\n');
                catch ME
                    try
                        saveas(gcf, '烟幕遮蔽示意图.fig');
                        fprintf('图片已保存为: 烟幕遮蔽示意图.fig\n');
                    catch
                        fprintf('图片保存失败: %s\n', ME.message);
                    end
                end
                
                hold off;
                
                plot_drawn = true;
            end
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