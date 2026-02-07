% 最简单的方法：明确告诉readtable第一行是表头
T = readtable('mydata.xlsx', 'Sheet', 'Sheet1', 'ReadVariableNames', true);

% 检查行数
fprintf('行数: %d\n', height(T));

% 如果是4行，直接添加E列
if height(T) == 4
    E = [10; 20; 30; 40];
    T.E = E;
    writetable(T, 'mydata.xlsx', 'Sheet', 'Sheet1');
    disp('成功!');
end