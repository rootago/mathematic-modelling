options = odeset('RelTol',1e-3,'AbsTol',1e-6);
[T,Y] = ode15s(@df,[0, 4*pi],[0,1,1],options)