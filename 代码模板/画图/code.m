%% 二维折线图
x = 0:0.1:10; 
y1 = sin(x);
y2 = cos(x);

figure;
plot(x, y1, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); hold on;
plot(x, y2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2); hold on;
legend('sin(x)','cos(x)');
title('三角函数对比','FontSize',14, 'FontName','SimSun');
grid on;
%% 散点图
x = randn(100,1);
y = randn(100,1);

figure;
scatter(x, y, 50, 'o','MarkerEdgeColor', 'r','MarkerFaceColor', 'b'); % 50是点大小
% scatter(x, y, 50, '*','MarkerEdgeColor', 'r'); % 50是点大小
xlabel('X');
ylabel('Y');
title('散点图示例');
%% 柱状图
data = [5 8 6 9 7];
figure;
bar(data);
xlabel('类别');
ylabel('数值');
title('柱状图示例');
%% 直方图
data = randn(1000,1); % 生成1000个正态分布随机数
figure;
histogram(data, 20); % 分20组
xlabel('数值');
ylabel('频数');
title('直方图');
%% 三维曲面图
[x, y] = meshgrid(-3:0.1:3, -3:0.1:3);
z = sin(x).*cos(y);

figure;
surf(x, y, z);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('三维曲面图','FontSize',14, 'FontName', 'SimSun');
shading interp; % 光滑
colorbar;       % 加颜色条
%% 
% 1. 生成网格
[x, y] = meshgrid(-3:0.1:3, -3:0.1:3);

% 2. 定义函数 z = sin(x)*cos(y)
z = sin(x).*cos(y);

% 3. 绘制三维曲面
figure;
surf(x, y, z);

% 4. 加标签和标题（宋体，字号14）
xlabel('X','FontName','Times New Roman','FontSize',14);
ylabel('Y','FontName','Times New Roman','FontSize',14);
zlabel('Z','FontName','Times New Roman','FontSize',14);
title('三维曲面图','FontName','SimSun','FontSize',14);

% 5. 美化
shading interp;   % 平滑过渡

pos = [0 0.3 1];
colors = [0 0.4627 0.6392; 0.4666 0.8 0.8157; 0.7882 0.9098 0.9020];
myMap = interp1(pos, colors, linspace(0,1,256));
colormap(myMap); 
colorbar;         % 颜色条

% 6. 导出图片
saveas(gcf, 'surface_plot.png');  % 保存为 PNG
