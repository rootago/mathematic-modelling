function points = generate_points()
    % 圆柱参数
    center = [0,200,0];   % 底面圆心
    R = 7;                % 半径
    H = 10; 
    
    % 采样精度
    n_r = 5;       % 半径方向点数
    n_theta = 36;  % 角度方向点数
    n_h = 11;      % 高度方向点数
    
    points = [];
    % 侧面采样
    thetas = linspace(0,2*pi,n_theta);
    hs = linspace(0,H,n_h);
    for theta = thetas
        for h = hs
            x = center(1) + R*cos(theta);
            y = center(2) + R*sin(theta);
            z = center(3) + h;
            points(end+1,:) = [x,y,z];
        end
    end
    % disp(length(points)); 396
    % 上底面采样 (z = z0 + H)
    rs = linspace(0,R,n_r);
    for r = rs
        for theta = thetas
            x = center(1) + r*cos(theta);
            y = center(2) + r*sin(theta);
            z = center(3) + H;
            points(end+1,:) = [x,y,z];
        end
    end
end

