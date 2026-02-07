function objectiveA = fun(x)
    t1 = x(1);
    t2 = x(2);
    v = x(3);
    theta = x(4);
    
    disp(x);
    objectiveA = 0;
    if t1 > t2
        objectiveA = objectiveA + 10000;
    end
    % 获取所有采样点
    points = generate_points();
    results = [];
    for i = 1:size(points,1)
        O = points(i,:);
        guesses = [4,8];
        solutions = zeros(1,4);
        try
            % options = optimoptions('fsolve','Display','off','TolX',1e-6);
            % for k = 1:2
            %     solution = solve(@(t) eq1(t1,t2,t,O,v,theta), guesses(k), options);
            %     solutions(k) = solution;
            % end
            syms t_sym
            eqn = eq1_sym(t1,t2,t_sym,O,v,theta) == 0;   % 构造方程
            disp('构造的方程:');
            disp(eqn);
            solution = solve(eqn, t_sym, 'Real', true);     % 求所有实数解
            disp(solution);
            solution = double(solution);
            disp(solution);
            % solutions = vpasolve(eqn, t_sym, [0, 20]);
            disp("solution: ");
            disp(solution);
            if ~isempty(solution) && ~any(isnan(solution))
                solution = solution(solution >= 0);
                if ~isempty(solution)
                    results = [results; solution(:)];
                end
            end
        catch ME
            disp('错误信息:');
            disp(ME.message);
            disp('错误标识:');
            disp(ME.identifier);
            continue
        end
        % if ~solution_found
            % solution = [0, 0];
        % end
        % disp("solution: ")
        % disp(solution);
    end
    % fprintf('当前x = [%s], 当前最大solution = %.4f\n', num2str(x), max(results));
    results = results(~isnan(results));
    disp("results: ");
    disp(results);

    if isempty(results)
        objectiveA = 100000; 
    else
        objectiveA = max(results);
    end

    disp("objectiveA: ");
    disp(objectiveA);
end
