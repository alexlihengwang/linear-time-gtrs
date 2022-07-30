% grad_descent_JL.m
% The saddle-point-based gradient descent method for JL19
function [x, x_old, out] = grad_descent_JL(A0, b0, c0, A1, b1, c1, A_0, b_0, c_0, A_1, b_1, c_1, opts)
    if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
    if ~isfield(opts, 'epsilon'); opts.epsilon = 1e-14; end
    L = opts.L * 1.1;   % this algorithm behaves poorly when L is too close to A_0 and A_1
    termination_criteria = opts.termination_criteria;

    n = length(b0);
    x = zeros(n,1);

    out = struct();
    epsilon = opts.epsilon;
    maxit = opts.maxit;
    m = 10;                                     % how often to log output
    out.conv = zeros(3, ceil(maxit / m) + 1);   % convergence data, times and values

    for t = 1 : maxit
        if mod(t, m) == 1
            tic;
        end
        x_old = x;
        f0_old = x_old' * A0 * x_old + 2 * b0' * x_old + c0;
        f1_old = x_old' * A1 * x_old + 2 * b1' * x_old + c1;

        if abs(f0_old - f1_old)/(abs(f0_old) + abs(f1_old)) < opts.epsilon
            g0 = 2 * (A0 * x_old + b0);
            g1 = 2 * (A1 * x_old + b1);

            alpha = - g0' * (g1 - g0) / ((g1 - g0)' * (g1 - g0));
            alpha = max(alpha, 0);
            alpha = min(alpha, 1);

            d = - ((1 - alpha) * g0 + alpha * g1);
            beta = 1 / L;
        else
            if f0_old > f1_old
                d = -2 * (A0 * x_old + b0);
            else
                d = -2 * (A1 * x_old + b1);
            end
            
            [beta1, beta2, success] = solve_quad(d' * (A0 - A1) * d, dot((A0-A1) * x_old + (b0 - b1), d), f0_old - f1_old);
                
            if success && 0 <= beta1 && beta1 <= 1 / L
                beta = beta1;
            elseif success && 0 <= beta2 && beta2 <= 1 / L
                beta = beta2;
            else
                beta = 1 / L;
            end
        end
            
        x_new = x_old + beta * d;
        x = x_new;
                
        if termination_criteria(A0, b0, c0, A1, b1, c1, L, x_new, x_old, x_old, epsilon)
            out.success = true;
            out.itr = t;
            if t <= m; k = 0; end
            out.conv(1, k+1) = toc;
            out.conv(2, k+1) = max(quad_eval(A0, b0, c0, x), quad_eval(A1, b1, c1, x));
            x_r = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, opts.v_minus, opts.v_plus);
            out.conv(3, k+1) = quad_eval(A_0, b_0, c_0, x_r);
            out.conv = out.conv(:, 1:k+1);
            return;
        end
        if mod(t, m) == 0
            k = t / m;
            out.conv(1, k) = toc;
            out.conv(2, k) = max(quad_eval(A0, b0, c0, x), quad_eval(A1, b1, c1, x));
            x_r = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, opts.v_minus, opts.v_plus);
            out.conv(3, k) = quad_eval(A_0, b_0, c_0, x_r);
        end
    end
    out.success = false;
    out.conv = out.conv(:, 1:k);
    out.itr = t;
    fprintf('termination criteria not met\n');
end


function [sol1, sol2, flag] = solve_quad(a, b, c)
    if a == 0
        error('Quadratic term == 0 in solve_quad');
    end
    
    b_norm = b / a;
    c_norm = c / a;
    if b_norm^2 - c_norm < 0
        flag = false;
        sol1 = [];
        sol2 = [];
    else
        disc = sqrt(b_norm^2 - c_norm);
        flag = true;
        sol1 = - b_norm - disc;
        sol2 = - b_norm + disc;
    end
end
