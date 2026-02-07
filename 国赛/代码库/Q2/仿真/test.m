t1 = 1.5;
t2 = 3.6;
v = 120;
theta = pi;
t = 9.4;
APos0 = [20000, 0, 2000];
alpha1 = atan(APos0(3)/APos0(1));
g = 9.8;
OPos = [0,207,0];
    
APos = [20000-300*t*cos(alpha1),0,2000-300*t*sin(alpha1)];
 % fprintf('导弹位置：[%f] ', APos);
 % 计算烟幕弹位           
 DPos = [17800 + v*(t1+t2)*cos(theta), v*(t1+t2)*sin(theta), 1800 - 0.5*g*t2^2 - 3*t];
            % fprintf('烟幕弹位置：[%f] ', DPos);
% 计算距离
OA = APos - OPos;
OD = DPos - OPos;
% fprintf('OA：[%f] ', OA);
% fprintf('OD：[%f] \n', OD);
distance1 = norm(cross(OA,OD))/norm(OA);
distance2 = norm(APos - DPos);
fprintf('norm(cross(OA,OD))：[%f]\n ', norm(cross(OA,OD)));
fprintf('distance1：[%f]\n ', distance1);
fprintf('distance2：[%f]\n ', distance2);
