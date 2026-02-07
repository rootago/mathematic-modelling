%% 染色体进行选择
function ret=Select(individuals,sizepop)

% individuals input  : 种群信息
% sizepop     input  : 种群规模
% ret         output : 更新种群
%根据个体适应度值进行排序
fitness1=10./individuals.fitness;
 
sumfitness=sum(fitness1);
sumf=fitness1./sumfitness;
index=[];
for i=1:sizepop   %转sizepop次轮盘
    pick=rand;
    while pick==0   
        pick=rand;       
    end
    for j=1:sizepop   
        pick=pick-sumf(j);       
        if pick<0       
            index=[index j];           
            break;  %寻找落入的区间，此次转轮盘选中了染色体i，注意：在转sizepop次轮盘的过程中，有可能会重复选择某些染色体
        end
    end
end
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;
