function Objective = fun(x)
    t1 = x(1);
    t2 = x(2);
    v = x(3);
    theta = x(4);
    
    disp(x);
    Objective = 0;

    [x1,x2] = eq1(t1,t2,v,theta);
    [X1,X2] = eq2(t1,t2,v,theta);
    
    Objective = max(x1,min(x2,X2));
    disp(Objective);
end
