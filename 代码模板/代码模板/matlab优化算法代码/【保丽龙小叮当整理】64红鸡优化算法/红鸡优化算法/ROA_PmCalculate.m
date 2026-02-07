function [ SC ,UC , D ] = ROA_PmCalculate (VarSize,MaxIt,it )

D(it) = (exp(it/(MaxIt))-(it/MaxIt))^-10;
D=D(it);

R = rand ;


if R <= 0.5
    SC = (rand+randi([1 2]))*rand(VarSize);
    UC = (rand+randi([1 3]))*rand(VarSize);
else
    SC = rand(VarSize);
    UC = (rand+randi([1 2]))*rand(VarSize);
end

end



