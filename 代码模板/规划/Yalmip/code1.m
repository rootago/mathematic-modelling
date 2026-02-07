x = sdpvar(1);
y = sdpvar(1);

constraints = [x+y<=2,2*x+3*y<=5,x>=0,y>=0]

objective = 3*x+4*y;

options = sdpsettings('solver','gurobi')

result = optimize(constraints,-objective)

if result.problem == 0
    value(x),value(y),value(objective)
else
    disp(result.info)
end