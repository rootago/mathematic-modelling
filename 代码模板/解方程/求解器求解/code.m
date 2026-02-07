% 定义决策变量
x = sdpvar(1,1,'full');
y = sdpvar(1,1,'full');

% 定义目标函数
Objective = x + y;

% 添加约束
Constraints = [];
% 方程1
Constraints = [Constraints;x+y==3];
% 方程2
Constraints = [Constraints;2*x-y==0];

% 设置求解器为 Gurobi
options = sdpsettings('solver','gurobi','verbose',1);

% 优化
optimize(Constraints, Objective, options);

% 输出结果
x_val = value(x);
y_val = value(y);

disp(['x = ', num2str(x_val)]);
disp(['y = ', num2str(y_val)]);