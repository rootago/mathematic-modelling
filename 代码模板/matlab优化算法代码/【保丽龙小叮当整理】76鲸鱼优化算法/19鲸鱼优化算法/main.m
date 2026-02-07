%% 相关代码由哔哩哔哩up主：保丽龙小叮当整理
%% 出现任何学术不端行为与本人无关，代码仅供参考学习，如有侵权请后台私信
%% 参考相关教材、网络博客和公开论文
%% 数据准备
%% 1清空环境变量

clear all 
clc
SearchAgents_no=30; % Number of search agents

Function_name='F1'; 

Max_iteration=500; % Maximum numbef of iterations

% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);

[Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

figure('Position',[269   240   660   290])
%Draw search space
subplot(1,2,1);
func_plot(Function_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

%Draw objective space
subplot(1,2,2);
semilogy(WOA_cg_curve,'Color','r')
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');

axis tight
grid on
box on

display(['The best solution obtained by WOA is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by WOA is : ', num2str(Best_score)]);

        



