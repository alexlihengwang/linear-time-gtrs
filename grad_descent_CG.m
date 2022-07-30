% grad_descent_CG.m
% Conjugate gradient descent method to solve Ax + b = 0.
% Returns x with ||Ax + b|| < epsilon
function [x, out] = grad_descent_CG(A, b, epsilon, opts)
    n = length(b);
    x = zeros(n, 1);
    out = struct();
    out.flag = 1;
    if nargin <= 3; opts = struct(); opts.log = 0; end
    if opts.log == 1
        % log the time and error for convergence plots
        out.conv = zeros(2, n);
        A_0 = opts.A_0; b_0 = opts.b_0; c_0 = opts.c_0; m = 5;
    end

    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:n
        if opts.log == 1
            if mod(i, m) == 1; tic; end
        end

        % Conjugate gradient descent
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        
        if sqrt(rsnew) < epsilon
            out.flag = 0;
            if opts.log == 1 && i <= m; k = 0; end
            if opts.log == 1
                out.conv(1, k+1) = toc;
                out.conv(2, k+1) = quad_eval(A_0, b_0, c_0, x);
                out.conv = out.conv(:, 1:k+1);
                out.itr = i;
            end
            return;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
        if opts.log == 1 && mod(i, m) == 0
            k = i / m;
            out.conv(2, k) = quad_eval(A_0, b_0, c_0, x);
            out.conv(1, k) = toc;
        end
    end
    out.itr = i;
end
