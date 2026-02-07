function [x1,x2] = eq2(t1,t2,v,theta)

    APos0 = [20000, 0, 2000];
    BPos0 = [17800, 0, 1800];
    AVel = 300;
    alpha1 = atan(APos0(3)/sqrt(APos0(1)^2 + APos0(2)^2));
    ADir = [-cos(alpha1), 0, -sin(alpha1)];
    g = 9.8;

    x1 = sdpvar(1,1,'full');
    x2 = sdpvar(1,1,'full');
    
    % x1对应的位置计算
    APos1 = APos0 + (x1+t1+t2)*ADir*AVel;
    DPos1 = [BPos0(1) + v*(t1+t2)*cos(theta),...
            BPos0(2) + v*(t1+t2)*sin(theta),...
            BPos0(3) - 0.5*g*t2^2 - 3*x1];

    % x2对应的位置计算
    APos2 = APos0 + (x2+t1+t2)*ADir*AVel;
    DPos2 = [BPos0(1) + v*(t1+t2)*cos(theta),...
             BPos0(2) + v*(t1+t2)*sin(theta),...
             BPos0(3) - 0.5*g*t2^2 - 3*x2];

    % 计算x1对应的距离
    distance1 = norm(DPos1-APos1);
    
    % 计算x2对应的距离
    distance2 = norm(DPos1-APos1);

    Objective = x1 + x2;
    Constraints = [];

    % 约束条件
    Constraints = [Constraints, x1 <= x2 + 1e-6];
    Constraints = [Constraints, distance1 == 10];  % x1满足距离约束
    Constraints = [Constraints, distance2 == 10];  % x2也满足距离约束

    % 设置求解器为 ipopt
    options = sdpsettings('solver','ipopt','verbose',1);

    % 优化
    optimize(Constraints, Objective, options);

    % 返回求解后的x1,x2值
    x1 = value(x1);
    x2 = value(x2);
   
end

