%% 红鸡优化算法---群文件自行下载
%% MATLAB重磅源码 打造一站式学习交流基地
%% 附送入群资格！！含上千份源码
%% 分享热爱，热爱分享！
%% 相关代码由管理员整理or优化or修改or原创
%% 代码仅供参考学习，超详细注释，如有侵权请后台私信
%% 参考相关教材、网络博客和公开论文
%% 群内资料超多，涵盖matlab、python入门资料、群智能优化算法、控制理论、高效工具等方面。
%% 提倡知识付费！白菜价code！伸手党勿扰！不包含讲解!
%% 感谢大家的关注，祝一切顺利！
%% 清除变量
clearvars
close all
clc
%% Initialization

D=10;
Xmin=-100;
Xmax=100;
pop_size=100;
iter_max=1000;
fhd=str2func('cec18_func');
%%
for i=1:3

    BestChart=[];

    func_num=i
    if(i==2)
        continue;
    end


    [gbest,Fitness,BestCh]= ROA(fhd,D,pop_size,iter_max,Xmin,Xmax,func_num);

    BestChart=[BestChart ,BestCh];
    disp(['The Best fitness of F', num2str(i),' is : ',num2str(Fitness)]);


    %% Plot
    figure
    semilogy(BestChart, 'LineWidth',1.5);
    legend('ROA' );
    xlabel('Iteration');
    ylabel(['Average Best solotion for Function F',num2str(i)]);

end
disp("===========================")
disp('欢迎进入小叮当的个人店铺')
disp('致力打造优化算法+高校科研+高效生活的学习聚集地')
disp('三个以上代码私聊即可获得 八折 优惠')
web('https://mbd.pub/o/xdd666')



