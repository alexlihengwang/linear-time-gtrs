% grad_descent_WK.m
% The Nesterov's accelerated gradient method used in WK20 and WLK21
function [x, y, out] = grad_descent_WK(A0, b0, c0, A1, b1, c1, A_0, b_0, c_0, A_1, b_1, c_1, opts)
    if ~isfield(opts, 'maxiter'); opts.maxiter = 100000; end
    if ~isfield(opts, 'epsilon'); opts.epsilon = 1e-8; end
    if ~isfield(opts, 'mu'); opts.mu = 0; end
    if ~isfield(opts, 'x_prev')
    opts.x_prev = zeros(length(b0),1);
    end
    if ~isfield(opts, 'y_prev'); opts.y_prev = opts.x_prev; end

    % The following algorithm uses the more standard notation
    % so that x^T Ai x + 2 bi^T x + ci are L-smooth, mu-strongly convex
    mu = 2 * opts.mu;
    L = 2 * opts.L;

    termination_criteria = opts.termination_criteria;
    x = opts.x_prev;
    y = opts.y_prev;

    qf = mu / L;
    alpha = 0.5 * (sqrt(qf) + 2 * (3 + qf) / (3 + sqrt(21 + 4 * qf)));

    out = struct();
    epsilon = opts.epsilon;
    maxiter = opts.maxiter;
    m = 10;                                     % how often to log output
    out.conv = zeros(4, ceil(maxiter / m) + 1); % convergence data, times and values

    for t = 1 : maxiter
        if mod(t, m) == 1
            tic;
        end
        alpha_new = (sqrt((alpha^2 - qf)^2 + 4 * alpha^2) - alpha^2 + qf) / 2;
        beta = alpha * (1 - alpha) / (alpha^2 + alpha_new);
        alpha = alpha_new;
        y_old = y;
        x_old = x;
        x_new = grad_map(L, 2 * (A0 * y + b0), quad_eval(A0, b0, c0, y), ...
            2 * (A1 * y + b1), quad_eval(A1, b1, c1, y), y);
        y = x_new + beta * (x_new - x);
        x = x_new;

        if mod(t, m) == 0
            k = t / m;
            out.conv(1, k) = toc;
            out.conv(2, k) = max(quad_eval(A0, b0, c0, x), quad_eval(A1, b1, c1, x));
            x_r = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, opts.v_minus, opts.v_plus);
%             out.conv(3, k) = quad_eval(A_0, b_0, c_0, x_r);
            out.conv(3, k) = quad_eval(A_0, b_0, c_0, x);
        end    

        if termination_criteria(A0, b0, c0, A1, b1, c1, L, x_new, y_old, x_old, epsilon)
            out.success = true;
            out.itr = t;
            if t <= m
                k = 0;
            end
            out.conv(1, k+1) = toc;
            out.conv(2, k+1) = max(quad_eval(A0, b0, c0, x), quad_eval(A1, b1, c1, x));
            x_r = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, opts.v_minus, opts.v_plus);
%             out.conv(3, k+1) = quad_eval(A_0, b_0, c_0, x_r);
            out.conv(3, k+1) = quad_eval(A_0, b_0, c_0, x);
            out.conv = out.conv(:, 1:k+1);
            return;
        end
    end

    out.success = false;
    out.conv = out.conv(:, 1:k);
    out.itr = t;
    fprintf('termination criteria not met\n');
end


function output = grad_map(L, g_0, c_0, g_1, c_1, y)
    % argmin_x (L/2) * |x - y|^2 + max_i <g_i, x - y> + c_i

    z_0 = y - g_0 / L;
    z_1 = y - g_1 / L;
    h_0 = c_0  - dot(g_0,g_0) / (2 * L);
    h_1 = c_1  - dot(g_1,g_1) / (2 * L);

    if z_0 == z_1
        output = z_0;
    else
        alpha_star = 0.5 - (h_0 - h_1) / (L * dot(z_0 - z_1, z_0 - z_1));
        alpha_star = max(alpha_star, 0);
        alpha_star = min(alpha_star, 1);
        output = z_0 + alpha_star * (z_1 - z_0);
    end
end
