% ode15i处理二阶导数DAE的降阶方法

clear; clc;

%% 原始方程
% x'' + 2*x' + x = cos(t)    (微分方程)
% x + y = sin(t)            (代数约束)

%% 降阶
% 引入新变量：v = x'
% 原系统变为：
% x' = v                    (定义关系)
% v' + 2*v + x = cos(t)     (二阶方程变一阶)
% x + y = sin(t)            (代数约束不变)

%% 1. 定义降阶后的DAE方程组
function res = dae(t, z, zp)
    
    x = z(1);   
    v = z(2);    
    y = z(3);  
    
    xp = zp(1); 
    vp = zp(2); 
    yp = zp(3); 
    
    res = zeros(3,1);
    res(1) = xp - v;                    % x' - v = 0 (定义关系)
    res(2) = vp + 2*v + x - cos(t);         % v' + 2*v + x - cos(t) = 0
    res(3) = x + y - sin(t);           % x + y - sin(t) = 0 (代数约束)
end

%% 2. 设置时间范围和初始条件
t0 = 0;
tf = 2*pi;
% tspan = linspace(t0, tf, 201);
tspan = linspace(t0, tf);

% 初始条件
x0 = 1;      
v0 = 0.5;
y0 = -x0;    

z0 = [x0; v0; y0];                  
zp0 = [v0; -2*v0 - x0 + 1; 1 - v0];  

%% 3. 求解降阶后的DAE
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t, z] = ode15i(@dae, tspan, z0, zp0, options);

% 提取结果
x = z(:,1);  
v = z(:,2);  
y = z(:,3); 

%% 4. 显示部分结果
disp(['时间点数量: ', num2str(length(t))]);
disp(' ');

% 显示所有时间点的结果
disp('所有时间点的数值解：');
disp('     时间t          x          v          y');
disp('-----------------------------------------------');
for i = 1:length(t)
    fprintf('%10.4f  %10.6f  %10.6f  %10.6f\n', t(i), x(i), v(i), y(i));
end

%% 5. 验证约束满足情况
constraint_error = abs(x + y - sin(t));
max_error = max(constraint_error);
disp(' ');
disp(['代数约束的最大误差: ', num2str(max_error)]);

%% 降阶的一般规律：
% 对于 n 阶导数的DAE：
% 1. 引入新变量：y1 = x, y2 = x', y3 = x'', ..., yn = x^(n-1)
% 2. 转换为 n 个一阶方程：
%    y1' = y2
%    y2' = y3
%    ...
%    yn-1' = yn  
%    yn' = f(t, y1, y2, ..., yn) (从原n阶方程得出)
% 3. 加上原有的代数约束