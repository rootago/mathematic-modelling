function [t1,t2] = eqn(t1, t2, v, theta, APos0, BPos0)
    % 参数
    AVel = 300;
    alpha1 = atan(APos0(3)/sqrt(APos0(1)^2 + APos0(2)^2));
    ADir = [-cos(alpha1), 0, -sin(alpha1)];
    g = 9.8;
    
    % 定义决策变量
    tA = sdpvar(1,1,'full');
    tB = sdpvar(1,1,'full');

    % 定义目标函数
    Objective = tA + tB;
    
    % 计算中间变量
    % 计算导弹位置
    APos1 = APos0 + (tA+t1+t2)*ADir*AVel;
    % 计算烟幕弹位置
    DPos1 = [BPos0(1) + v*(t1+t2)*cos(theta),...
            BPos0(2) + v*(t1+t2)*sin(theta),...
            BPos0(3) - 0.5*g*t2^2 - 3*tA];

     % 计算距离
     OA1 = APos - OPos;
     OD1 = DPos - OPos;
     distance1A = norm(cross(OA,OD))/norm(OA);
     distance2A = norm(APos - DPos);
    
    % 计算导弹位置
    APos2 = APos0 + (tB+t1+t2)*ADir*AVel;
    % 计算烟幕弹位置
    DPos2 = [BPos0(1) + v*(t1+t2)*cos(theta),...
            BPos0(2) + v*(t1+t2)*sin(theta),...
            BPos0(3) - 0.5*g*t2^2 - 3*tB];

     % 计算距离
     OA2 = APos - OPos;
     OD2 = DPos - OPos;
     distance1B = norm(cross(OA,OD))/norm(OA);
     distance2B = norm(APos - DPos);

    % 添加约束
    Constraints = [];
    Constraints = [Constraints;t1<t2];
    % 方程1
    Constraints = [Constraints;distance1A==0];
    Constraints = [Constraints;distance1B==0];
    % 方程2
    Constraints = [Constraints;distance2A==0];
    Constraints = [Constraints;distance2B==0];
    % 设置求解器为 Gurobi
    options = sdpsettings('solver','fmincon','verbose',1);
    
    % 优化
    optimize(Constraints, Objective, options);
    
    % 输出结果
    t1_val = value(x);
    t2_val = value(t2);
    
    disp(['x = ', num2str(x_val)]);
    disp(['y = ', num2str(y_val)]);
end

