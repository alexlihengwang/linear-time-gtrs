% gtrs_BTH14.m
% Implementation of the algorithm in the paper "Hidden conic quadratic representation of some nonconvex quadratic optimization problems" by A. Ben-Tal and D. den Hertog.
% Requirement: MOSEK installed
function [x_opt, fval, out] = gtrs_BTH14(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat)
    tic;
    clear prob;
    out = struct();

    % compute simultaneously-diagonalizing basis
    n = size(A_0,1);
    A_hat = A_0 + gamma_hat*A_1;

    [S, ~] = eigs(A_1, A_hat, n);
    out.t1_eigs = toc;

    tic;
    lambda = diag(S' * A_0 * S);
    theta = diag(S' * A_1 * S);
    alpha = S' * b_0;
    beta = S' * b_1;
    out.t2 = toc;

    % construct sparse matrix and vectors
    tic;
    i = [1:n, 1:n, (n+1):(2*n), (n+1):(2*n)]';
    j = [(n+1):(2*n), (2*n+1):(3*n), (n+1):(2*n), (3*n+1)*ones(1,n)]';
    v = [beta; -theta; ones(n,1); -theta];
    A = sparse(i, j, v, 2*n, 3*n+1);
    blc = [beta .* lambda - theta .* alpha; lambda];

    % SOCP reformulation with MOSEK
    [~, res] = mosekopt('symbcon');	% Specify the non-conic part of the problem.

    % variable: [t_1/2, ..., t_n/2, ..., ..., gamma]'
    prob.c   = [2*ones(1,n), zeros(1,2*n), -c_1]';
    prob.a   = A;	% linear constraints matrix
    prob.blc = blc;
    prob.buc = blc;
    prob.blx = [zeros(1,2*n), -inf(1,n), 0]';   % variable lower bounds
    prob.bux = inf(3*n+1,1);

    % n rotated SOCs
    prob.cones.type = zeros(n,1);
    prob.cones.sub = zeros(3*n,1);
    prob.cones.subptr = zeros(n,1);
    for i = 1:n
        prob.cones.type(i) = res.symbcon.MSK_CT_RQUAD;
        prob.cones.sub(3*i-2) = i; 
        prob.cones.sub(3*i-1) = i + n; 
        prob.cones.sub(3*i) = i + 2*n; 
        prob.cones.subptr(i) = 3*i - 2;
    end

    param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-12;
    param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-12;
    param.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-12;
    param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-12;
    param.MSK_DPAR_OPTIMIZER_MAX_TIME = 10000;

    out.t3 = toc;

    tic;
    % solve the SOCP reformulation using Mosek
    [~, res] = mosekopt('minimize echo(0) log(results/mosek_log.txt)', prob, param);
    % solution 
    fval = -res.sol.itr.pobjval + c_0;
    z = res.sol.itr.xx;
    gamma_star = z(3 * n + 1);
    x_opt = -(A_0 + gamma_star * A_1) \ (b_0 + gamma_star * b_1);
    out.fopt = quad_eval(A_0, b_0, c_0, x_opt);
    out.t_mosek = toc;

    out.x = x_opt;
    out.fval = fval;
    out.q1 = quad_eval(A_1, b_1, c_1, x_opt);
    out.time = out.t1_eigs + out.t2 + out.t3 + out.t_mosek;
end
