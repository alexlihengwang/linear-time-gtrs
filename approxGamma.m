% approxGamma.m
% Step 1.(b) of CRLeft (Algorithm 2) in WLK21.
% Generalized-eigenvalue-based replacement for ApproxGammaLeft (Algorithm 5) of CRLeft (Algorithm 2).
function [gamma, v] = approxGamma(A_0, A_1, gamma_hat, mu, xi, opt)
    n = size(A_0, 1);
    % parameters for eigifp
    opts = struct();
    opts.useprecon = 'NO';
    opts.disp = 0;
    opts.maxit = 1000;
    if isfield(opt, 'v0'); opts.V0 = opt.v0; end

    A_hat = A_0 + gamma_hat * A_1;
    % tol = min(mu / 4 / (1 + opt.zeta - gamma_hat + mu), mu * xi / (5 * (2*xi-mu)));
    tol = mu * xi^2 / (360 * opt.zeta^3);       % given in Section D
    tol = max(tol, 1e-10);
    opts.tol = tol;
    A_tilde = A_hat - 0.75 * mu * speye(n);

    if opt.plus == 1
        % CRRight
        [gamma, v] = eigifp(A_1, A_tilde, opts);
        gamma = gamma_hat - 1 / (gamma - opts.tol);
    else
        % CRLeft
        [gamma, v] = eigifp(-A_1, A_tilde, opts);
        gamma = gamma_hat + 1 / (gamma - opts.tol);
    end
end
