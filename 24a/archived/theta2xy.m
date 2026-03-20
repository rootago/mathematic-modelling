function [resx, resy] = theta2xy(theta)
    load('constants.mat');
    a = PITCH / (2 * PI);
    resx = a * theta .* cos(theta);
    resy = a * theta .* sin(theta);
end