function val = eq1_sym(t1,t2,t_sym,O,v,theta)
    
    uavPos = [17800, 0, 1800];
    missilePos = [20000, 0, 2000];
    alpha1 = atan(missilePos(3)/missilePos(1));
    alpha2 = atan(uavPos(3)/uavPos(1));
    g = 9.8;
    
    A = [20000 - (300*t_sym+300*t1+300*t2)*cos(alpha1), ...
         0, 2000 - (300*t_sym+300*t1+300*t2)*sin(alpha1)];
    D = [17800 + v*(t1+t2)*cos(theta), ...
         v*(t1+t2)*sin(theta), ...
         1800 - 0.5*g*t2^2 - 3*t_sym];
    
    OA = A - O;
    OD = D - O;
    
    cross_product = cross(OA, OD);
    val = sqrt(sum(cross_product.^2)) / sqrt(sum(OA.^2)) - 10;
end