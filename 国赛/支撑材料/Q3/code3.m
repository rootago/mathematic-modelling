clear;clc
% 初始点
x0 = [1.5,3.6,1.5,3.6,1.5,3.6,120,pi];
% 变量边界
lb = [0, 1.4, 0, 1.4, 0, 1.4, 70, 0];
ub = [7, 7, 7, 7, 7, 7, 140, pi];
nvars = 8;
rng(0);
% 设置选项
options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'PlotFcn', @pswplotbestf, ...
    'SwarmSize', 200, ...
    'MaxIterations', 200, ...
    'MaxStallIterations', 200, ...
    'FunctionTolerance', 1e-12, ...
    'UseParallel', true);
% 调用粒子群算法
[x,fval] = particleswarm(@fun, nvars, lb, ub, options);
disp('最优解：');
disp(x);
disp('最优目标值：');
disp(- fval);
% 计算三个烟幕弹的投放点位置
drop_pos = [];
for i = 1:2:5
    t = x(i);
    t1 = x(i);
    t2 = x(i+1);
    v = x(7);
    theta = x(8);
    g = 9.8;
    DPos = [17800 + v*(t1+t2)*cos(theta), v*(t1+t2)*sin(theta), 1800 - 0.5*g*t2^2 - 3*t];
    drop_pos = [drop_pos;DPos];
end
disp("烟幕弹投放点位置");
disp(drop_pos);
% 计算三个烟幕弹的投放点位置
explode_pos = [];
for i = 2:2:6
    t = x(i);
    t1 = x(i-1);
    t2 = x(i);
    v = x(7);
    theta = x(8);
    g = 9.8;
    DPos = [17800 + v*(t1+t2)*cos(theta), v*(t1+t2)*sin(theta), 1800 - 0.5*g*t2^2 - 3*t];
    explode_pos = [explode_pos;DPos];
end
disp("烟幕弹起爆点位置");
disp(explode_pos);
% 计算时间区间
final_result = -fun(x)
% 画图逻辑 - 从最优解x中获取时间
% x的前6个元素分别是t1_start, t1_end, t2_start, t2_end, t3_start, t3_end
t1_start = x(1); t1_end = x(2);
t2_start = x(3); t2_end = x(4);
t3_start = x(5); t3_end = x(6);
% 计算排序后的区间
times = [];
for i = 1:2:5
    t1 = x(i);
    t2 = x(i+1);
    v = x(7);
    theta = x(8);
    [t_start,t_end] = getTime(t1,t2,v,theta);
    times = [times;[t_start, t_end]];
end
% 排序后的区间
times_sorted = sortrows(times, 1);
% 输出排序后区间的时间长度
fprintf('排序后区间的时间长度：\n');
for i = 1:size(times_sorted, 1)
    duration = times_sorted(i, 2) - times_sorted(i, 1);
    fprintf('区间%d: [%.4f, %.4f] 长度: %.4f\n', i, times_sorted(i, 1), times_sorted(i, 2), duration);
end
% 创建图形
figure;
hold on;
% 画三个烟雾弹的原始区间（虚线）
plot([t1_start, t1_end], [1, 1], 'r--', 'LineWidth', 2);
plot([t2_start, t2_end], [2, 2], 'g--', 'LineWidth', 2);
plot([t3_start, t3_end], [3, 3], 'b--', 'LineWidth', 2);
% 画三个烟雾弹的排序后区间（实线）
plot([times_sorted(1,1), times_sorted(1,2)], [1, 1], 'r-', 'LineWidth', 3);
plot([times_sorted(2,1), times_sorted(2,2)], [2, 2], 'g-', 'LineWidth', 3);
plot([times_sorted(3,1), times_sorted(3,2)], [3, 3], 'b-', 'LineWidth', 3);
% 设置图形属性
xlabel('时间 t');
title('烟雾弹时间区间','FontSize',14);
grid on;
ylim([0.5, 3.5]);
set(gca, 'YTick', [1, 2, 3], 'YTickLabel', {'烟雾弹1', '烟雾弹2', '烟雾弹3'});
legend('原始区间1', '原始区间2', '原始区间3', '排序后区间1', '排序后区间2', '排序后区间3', 'Location', 'best');
hold of