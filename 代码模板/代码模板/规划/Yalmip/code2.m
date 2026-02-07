% 集合定义
I = 1:m;   % 行集
J = 1:n;   % 列集

% 参数 (需用户定义)
c = sdpvar(m,n,'full');  % 系数矩阵 
a = sdpvar(m,n,'full');  % 系数矩阵 (约束)
b = sdpvar(m,1);         % 右端常数

% 决策变量
x = sdpvar(m,n,'full');  % 连续变量
% x = binvar(m,n,'full'); % 0-1 变量
% x = intvar(m,n,'full'); % 整数变量

% 目标函数
Objective = sum(sum(c.*x));

% 约束
Constraints = [];
for i = I
    Constraints = [Constraints, sum(a(i,:).*x(i,:)) <= b(i)];
end
Constraints = [Constraints, x >= 0]; % 非负约束

% 求解
options = sdpsettings('solver','gurobi','verbose',1);
sol = optimize(Constraints, -Objective, options);

% 结果
if sol.problem == 0
    disp('Optimal solution found:');
    value(Objective)
    value(x)
else
    disp('Solver failed:');
    sol.info
end
