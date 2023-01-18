% gtrs_WK20.m
% Implementation of algorithm in the paper "The generalized trust region subproblem: solution complexity and convex hull results."
function [x, fval, out] = gtrs_WK20(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts)
    tic;
    % default parameters
    if ~isfield(opts, 'grad_alg'); opts.grad_alg = @grad_descent_WK; end
    if ~isfield(opts, 'termination_criteria'); opts.termination_criteria = @termination_criteria_WK; end
    if ~isfield(opts, 'opts_gamma')
        opts.opts_gamma = struct();
        opts_gamma = opts.opts_gamma;
        opts_gamma.tol = 1e-10;
        opts_gamma.useprecon = 'NO';
        opts_gamma.disp = 0;
        opts_gamma.maxit = 1000;
    end
    if ~isfield(opts, 'opts_L')
        opts.opts_L = struct();
        opts_L = opts.opts_L;
        opts_L.tol = 0.1;
        opts_L.useprecon = 'NO';
        opts_L.disp = 0;
        opts_L.maxit = 1000;
    end
    if ~isfield(opts, 'opts_grad')
        opts.opts_grad = struct();
    end

    out = struct();
        
    A_hat = A_0 + gamma_hat * A_1;
    
    % construct convex reformulation of the GTRS
    % compute gamma_minus & gamma_plus
    fprintf('\tComputing gamma_minus and gamma_plus\n');
    [gamma_minus, v_minus] = eigifp(-A_1, A_hat, opts_gamma);
    gamma_minus = gamma_hat + 1 / (gamma_minus - opts_gamma.tol);
    v_minus = v_minus / norm(v_minus);

    [gamma_plus, v_plus] = eigifp(A_1, A_hat, opts_gamma);
    gamma_plus = gamma_hat - 1 / (gamma_plus - opts_gamma.tol);
    v_plus = v_plus / norm(v_plus);
    out.time_eig = toc;
    fprintf('\tEigenvalue time, %f\n', out.time_eig);

    % Solve the convex reformulation
    % parameters for gradient descent
    tic;
    A_minus = A_0 + gamma_minus * A_1;
    b_minus = b_0 + gamma_minus * b_1;
    c_minus = c_0 + gamma_minus * c_1;
    A_plus = A_0 + gamma_plus * A_1;
    b_plus = b_0 + gamma_plus * b_1;
    c_plus = c_0 + gamma_plus * c_1;
    L = max(-eigifp(-A_minus, opts_L), -eigifp(-A_plus, opts_L)) + opts_L.tol;
    
    opts_grad = opts.opts_grad;
    opts_grad.L = L;
    opts_grad.termination_criteria = opts.termination_criteria;
    out.time_L = toc;
    opts_grad.v_plus = v_plus;
    opts_grad.v_minus = v_minus;

    % gradient descent
    fprintf('\tRunning gradient descent\n');
    [x, ~, out_grad] = opts.grad_alg(A_minus, b_minus, c_minus, ...
        A_plus, b_plus, c_plus, A_0, b_0, c_0, A_1, b_1, c_1, opts_grad);

    conv = out_grad.conv;
    out.time_log = conv(1, :);
    out.Fcvx = conv(2, :);
    out.Fq0 = conv(3, :);
    out.fval_cvx = out.Fcvx(end);
    out.time_grad = sum(conv(1, :));
    fprintf('\tGrad descent time, %f\n', out.time_grad);
    
    % rounding
    tic;
    fprintf('\tRounding\n');
    x = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, v_minus, v_plus);
    out.time_rounding = toc;
        
    fval = quad_eval(A_0, b_0, c_0, x);

    out.x = x;
    out.fval = fval;
    out.time = out.time_eig + out.time_grad + out.time_rounding + out.time_L;
    out.itr_grad = out_grad.itr;
    out.q1 = quad_eval(A_1, b_1, c_1, x);
end
