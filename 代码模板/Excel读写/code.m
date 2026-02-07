%% write 
A = [1 2 3; 4 5 6; 7 8 9];          % 一个 3x3 矩阵
writematrix(A, 'mydata.xlsx');       % 写入 Excel 文件（自动创建 Sheet1）
writematrix(A, 'mydata.xlsx','Sheet','Sheet3');
%% read (新版)
B = readmatrix('mydata.xlsx');       % 读取 Excel 文件中的数值
disp(B);
%% read
% 读原始表头（不修改）
T = readtable('mydata.xlsx', 'Sheet', 'Sheet1');
disp(T);
% 显示原始列名
T.Properties.VariableNames
%% D 
T = readtable('mydata.xlsx', 'Sheet', 'Sheet1', 'ReadVariableNames', true);

G = [11; 12; 13; 14];
T.("心情") = G;
writetable(T, 'mydata.xlsx', 'Sheet', 'Sheet1');
