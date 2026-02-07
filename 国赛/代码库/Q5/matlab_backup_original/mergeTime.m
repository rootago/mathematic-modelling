function merged = mergeTime(times)
    fprintf('DEBUG: mergeTime被调用，输入区间数量：%d\n', size(times,1));
    times = times(times(:,1) > 0, :);

    if isempty(times)
        disp("无有效区间");
        merged = [];
        return;
    end
    
    % 按开始时间排序
    times = sortrows(times, 1);
    disp("排序后的区间");
    disp(times);

    % 合并区间
    merged = times(1,:);

    for i = 2:size(times,1)
        curr = times(i,:);
        last = merged(end,:);

        if curr(1) <= last(2)
            merged(end,2) = max(curr(2),last(2));
        else
            merged = [merged; curr];
        end
    end
    
    disp("合并后的区间");
    disp(merged);
end

