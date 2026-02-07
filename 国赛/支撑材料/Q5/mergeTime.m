function merged = mergeTime(times)
    times = times(times(:,1) > 0, :);

    if isempty(times)
        disp("无有效区间");
        merged = [];
        return;
    end
    
    % 按开始时间排序
    times = sortrows(times, 1);

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
    
end

