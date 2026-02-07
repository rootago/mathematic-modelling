options = odeset('RelTol',1e-3,'AbsTol',1e-6);
[T,Y] = ode45(@df,[1, 2],1/2,options)