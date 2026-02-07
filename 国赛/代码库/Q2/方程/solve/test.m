function debug_vpasolve()
    % 测试简单案例
    syms x
    
    disp('=== 测试1: 简单线性方程 ===');
    try
        sol1 = vpasolve(x - 5 == 0, x);
        disp(['简单方程解: ', char(sol1)]);
        disp(['解的类型: ', class(sol1)]);
    catch ME
        disp(['简单方程失败: ', ME.message]);
    end
    
    disp('=== 测试2: 简单非线性方程 ===');
    try
        sol2 = vpasolve(x^2 - 4 == 0, x, [0, 5]);
        disp(['非线性方程解: ', char(sol2)]);
        disp(['解的类型: ', class(sol2)]);
    catch ME
        disp(['非线性方程失败: ', ME.message]);
    end
    
    disp('=== 测试3: 包含sqrt的方程 ===');
    try
        sol3 = vpasolve(sqrt(x) - 2 == 0, x);
        disp(['sqrt方程解: ', char(sol3)]);
        disp(['解的类型: ', class(sol3)]);
    catch ME
        disp(['sqrt方程失败: ', ME.message]);
    end
end

debug_vpasolve();