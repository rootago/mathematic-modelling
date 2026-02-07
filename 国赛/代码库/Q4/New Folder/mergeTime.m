function total_time = mergeTime(times)
    
    total_time  = 0;
    
    times = sortrows(times, 1);
    disp("排序后的区间");
    disp(times);
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
    
    for i = 1:size(merged,1) 
        curr = merged(i,:);
        t_start = curr(1);
        t_end = curr(2);
        total_time = total_time + t_end - t_start;
    end
    disp("合并后的区间");
    disp(merged);
end

