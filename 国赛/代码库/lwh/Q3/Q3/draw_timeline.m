clear;clc

drone_data = [
    0.7293,    0.3289,    1.058172,    5.858172;   % 无人机1
    3.165,     6.94,      10.205,      14.605;     % 无人机2  
    40,        10,        52.213,      57.613      % 无人机3
];

% 提取时间数据
drop_time = drone_data(:, 1);      % 投放时间
explode_time = drone_data(:, 2);   % 起爆时间
shield_start = drone_data(:, 3);   % 遮蔽开始时间
shield_end = drone_data(:, 4);     % 遮蔽结束时间

% 创建图形
figure;
hold on;

% 画三架无人机的虚线区间（投放时间到起爆时间）
plot([drop_time(1), explode_time(1)], [1, 1], 'r--', 'LineWidth', 2);
plot([drop_time(2), explode_time(2)], [2, 2], 'g--', 'LineWidth', 2);
plot([drop_time(3), explode_time(3)], [3, 3], 'b--', 'LineWidth', 2);

% 画三架无人机的实线区间（遮蔽开始到遮蔽结束时间）
plot([shield_start(1), shield_end(1)], [1, 1], 'r-', 'LineWidth', 3);
plot([shield_start(2), shield_end(2)], [2, 2], 'g-', 'LineWidth', 3);
plot([shield_start(3), shield_end(3)], [3, 3], 'b-', 'LineWidth', 3);

% 设置图形属性
xlabel('时间 t');
title('无人机时间区间','FontSize',14);
grid on;
ylim([0.5, 3.5]);
set(gca, 'YTick', [1, 2, 3], 'YTickLabel', {'无人机1', '无人机2', '无人机3'});
legend('投放-起爆1', '投放-起爆2', '投放-起爆3', '遮蔽区间1', '遮蔽区间2', '遮蔽区间3', 'Location', 'best');
hold off;