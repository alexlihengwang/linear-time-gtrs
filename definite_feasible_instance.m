% definite_feasible_instance.m
% Generate random feasible instances for the GTRS as detailed in Section 4.2
function [A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, zeta, opt, x_opt] = definite_feasible_instance(n, xi, density, mu_opt)
    % generate A(gamma_hat)
    A_hat = sprandsym(n, density);
    % parameters for eigs
    tol = 1e-2;
    opts = struct();
    opts.disp = 0;
    opts.tol = tol;
    % compute approximate smallest and largest eigenvalue of A_hat
    temp1 = eigs(A_hat, 1, 'sr', opts) - tol;
    temp2 = eigs(A_hat, 1, 'lr', opts) + tol;
    % scale A_hat such that xi < ||A_hat||_2 < 1 + xi.
    A_hat = (1 / (temp2 - temp1)) * A_hat + (xi - temp1 / (temp2 - temp1)) * speye(n);

    % generate A_0
    A_0 = sprandsym(n, density);
    A_0 = A_0 / abs(eigs(A_0, 1, 'lm', opts) + tol);

    % compute gamma_hat & A_1
    gamma_hat = abs(eigs(A_hat - A_0,1,'lm', opts) + tol);
    A_1 = (A_hat - A_0) / gamma_hat;

    zeta = -1 / (eigs(A_1, 1, 'sr', opts) + tol);
    
    % generate b_0 & b_1.
    b_0 = rand(n, 1);
    b_0 = b_0 / norm(b_0);
    b_1 = rand(n, 1);
    b_1 = b_1 / norm(b_1);  
    c_0 = 0;

    % compute gamma_star
    opts_gamma = struct();
    opts_gamma.useprecon = 'NO';
    opts_gamma.disp = 0;
    n = size(A_0, 1);
    A_tilde = A_hat - mu_opt * speye(n);

    % randomly set gamma_star on the left/right of gamma_hat
    rand_num = randi(2);
    if rand_num == 1
        % set gamma_star on the left of gamma_hat
        [gamma, ~] = eigifp(-A_1, A_tilde, opts_gamma);
        % the error of gamma output by eigifp (given by the eigifp source code)
        tol = 10 * eps * sqrt(n) * (norm(A_1, 1) + gamma * norm(A_tilde, 1));     
        gamma_star = gamma_hat + 1 / (gamma - tol);
    else
        % set gamma_star on the right of gamma_hat
        [gamma, ~] = eigifp(A_1, A_tilde, opts_gamma);
        tol = 10 * eps * sqrt(n) * (norm(A_1, 1) + gamma * norm(A_tilde, 1));
        gamma_star = gamma_hat - 1 / (gamma - tol);
    end
    
    % compute the optimizer x_opt using conjugate gradient method
    A = A_0 + gamma_star * A_1;
    b = b_0 + gamma_star * b_1;
    [x_opt, ~] = grad_descent_CG(A, -b, 1e-16);
    
    % compute c_1 so that q_1(x_opt) = 0.
    c_1 = - (x_opt' * A_1 * x_opt + 2 * b_1' * x_opt);
        
    % scale c_1 if necessary
    if abs(c_1) >= 1
        scale = sqrt(abs(c_1));
        b_0 = b_0 / scale;
        b_1 = b_1 / scale;
        c_1 = c_1 / abs(c_1);
        x_opt = x_opt / scale;
    end
    opt = quad_eval(A_0, b_0, c_0, x_opt);
end
