% gtrs_AN19.m
% implementation of the algorithm in the paper "Eigenvalue-based algorithm and analysis for nonconvex QCQP with one constraint." by S. Adachi and Y. Nakatsukasa.
function [x_opt, fval, out] = gtrs_AN19(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts)
tic;
if ~isfield(opts, 'log'); opts.log = 0; end     % if keeps track of the convergence process
opts_eig = struct();
opts_eig.disp = 0;
opts_eig.fail = 'keep';

% tolerance for the eigenvalue solver
if isfield(opts, 'tol')
    opts_eig.tol = opts.tol;
else
    opts_eig.tol = 1e-14;
end
out = struct();

% construct sparse matrices M0 & M1
n = size(A_0,1);
M0 = spalloc(2 * n + 1, 2 * n + 1, 1 + 4 * n + nnz(A_1) + 2 * nnz(A_0));
M1 = spalloc(2 * n + 1, 2 * n + 1, 2 * n + nnz(A_1));
M0(1:1,1:1) = c_1;
M0(1:1, 2:n + 1) = b_1';
M0(2:n + 1, 1:1) = b_1;
M0(1:1, n + 2: 2 * n + 1) = -b_0';
M0(n + 2: 2 * n + 1, 1:1) = -b_0;
M0(2:n + 1, 2:n + 1) = A_1;
M0(2:n + 1, n + 2: 2 * n + 1) = -A_0;
M0(n + 2: 2 * n + 1, 2:n + 1) = -A_0;

M1(1:1, n + 2:2 * n + 1) = -b_1';
M1(n + 2:2 * n + 1, 1:1) = -b_1;
M1(2:n + 1, n + 2: 2 * n + 1) = -A_1;
M1(n + 2: 2 * n + 1, 2:n + 1) = -A_1;

M_hat = M0 + gamma_hat * M1;
x = -(A_0 + gamma_hat * A_1) \ (b_0 + gamma_hat * b_1);
g_x = quad_eval(A_1, b_1, c_1, x);

out.t1 = toc;

if g_x > 0
    % largest real eigenvalue
    [z, xi, out_eigs] = eigs_new(-M1, M_hat, 1, 'lr', opts_eig);
    gamma_star = gamma_hat + 1 / (xi + opts_eig.tol);  
elseif g_x < 0
    % smallest eigenvalue
    [z, xi, out_eigs] = eigs_new(-M1, M_hat, 1, 'sr', opts_eig);
    if gamma_hat == 0 || xi >= -1 / gamma_hat
        gamma_star = 0;
        x_opt = -A_0 \ b_0;
    else
        gamma_star = gamma_hat + 1 / (xi - opts_eig.tol);
    end  
else
    gamma_star = gamma_hat;
    x_opt = -(A_0 + gamma_hat * A_1) \ (b_0 + gamma_hat * b_1);
end

itr = -1;
if g_x ~= 0
    itr = out_eigs.itr;
    V_log = out_eigs.V_log;
    d_log = out_eigs.d_log;
end

tic;
n = size(A_0,1);
theta = z(1);
y1 =  z(2:(n+1));
if gamma_star > 0 && quad_eval(A_1, b_1, c_1, -(A_0 + gamma_hat * A_1) \ (b_0 + gamma_hat * b_1)) ~= 0
    if theta ~= 0
        x_opt = y1 / theta;
    else
        V = null(A_0 + gamma_star*A_1);
        % arbitary positive number
        alpha = 1;
        A_tilde = sparse(n,n);
        for k = 1:size(V,2)
            A_tilde = A_tilde + A_1*V(:,k)*V(:,k)'*A_1;
        end
        A_tilde = A_0 + gamma_star*A_1 + alpha*A_tilde;
        a_tilde = sparse(n,1);
        for k = 1:size(V,2)
            a_tilde = a_tilde + A_1*V(:,k)*V(:,k)'*b_1;
        end
        a_tilde = b_0 + gamma_star*b_1 + alpha*a_tilde;
        w_star = -A_tilde \ a_tilde;        
        v1 = V(:,1);
        t = sqrt( -quad_eval(A_1, b_1, c_1, w_star) / v1'*A_1*v1 );
        x_opt = w_star + t*v1;
    end
end

% Newton refinement process
jj = 0;
q1 = quad_eval(A_1, b_1, c_1, x_opt);
while jj < 20 && (q1 < -1e-12 || q1 > 0)
    delta_hat = 0.5 * q1 * (A_1 * x_opt + b_1) / norm(A_1 * x_opt + b_1)^2;
    x_opt = x_opt - delta_hat;  
    q1 = quad_eval(A_1, b_1, c_1, x_opt);
    jj = jj + 1;
end

out.x = x_opt;
fval = quad_eval(A_0, b_0, c_0, x_opt);
out.fval = fval;
out.q1 = quad_eval(A_1, b_1, c_1, x_opt);
out.t2 = out_eigs.time;
out.t3 = toc;
out.time = out.t1 + out.t2 + out.t3;
out.itr = itr;
% algorithm ends here

% Below keeps track of the convergence process for the plots
if opts.log == 1
    Fval = [];
    for ii = 1:itr
        z = V_log(:, ii);
        xi = d_log(ii);
        if g_x > 0
            gamma_star = gamma_hat + 1 / (xi + opts_eig.tol);  
        elseif g_x < 0
            if gamma_hat == 0 || xi >= -1/gamma_hat
                gamma_star = 0;
                x_opt = -A_0 \ b_0;
            else
                gamma_star = gamma_hat + 1 / (xi - opts_eig.tol);
            end  
        else
            gamma_star = gamma_hat;
            x_opt = -(A_0 + gamma_hat*A_1) \ (b_0 + gamma_hat*b_1);
        end
        
        theta = z(1);
        y1 =  z(2:(n+1));
   
        if gamma_star > 0 && quad_eval(A_1, b_1, c_1, -(A_0 + gamma_hat * A_1) \ (b_0 + gamma_hat * b_1)) ~= 0
            if theta ~= 0
                x_opt = y1 / theta;
            else
                V = null(A_0 + gamma_star*A_1);
                alpha = 1;
                A_tilde = sparse(n,n);
                for k = 1:size(V,2)
                    A_tilde = A_tilde + A_1*V(:,k)*V(:,k)'*A_1;
                end
                A_tilde = A_0 + gamma_star*A_1 + alpha*A_tilde;
                a_tilde = sparse(n,1);
                for k = 1:size(V,2)
                    a_tilde = a_tilde + A_1*V(:,k)*V(:,k)'*b_1;
                end
                a_tilde = b_0 + gamma_star*b_1 + alpha*a_tilde;

                w_star = -A_tilde \ a_tilde;
                v1 = V(:,1);
                t = sqrt( -quad_eval(A_1, b_1, c_1, w_star) / v1'*A_1*v1 );
                x_opt = w_star + t*v1;
            end
        end
        
        % Newton refinement process
        jj = 0;
        q1 = quad_eval(A_1, b_1, c_1, x_opt);
        while jj < 20 && (q1 < -1e-12 || q1 > 0)
            q1_old = q1;
            delta_hat = 0.5 * q1 * (A_1 * x_opt + b_1) / norm(A_1 * x_opt + b_1)^2;
            x_opt = x_opt - delta_hat;  
            q1 = quad_eval(A_1, b_1, c_1, x_opt);
            if q1 < 0 && q1 > q1_old
                break
            end
            jj = jj + 1;
        end
        Fval = [Fval, quad_eval(A_0, b_0, c_0, x_opt)];
    end
    out.Fval = Fval;
    out.time_log = out.t1 + out_eigs.time_log;
end
end
