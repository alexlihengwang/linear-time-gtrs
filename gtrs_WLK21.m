% gtrs_WLK21.m
% Implementation of WLK21 as proposed in the paper.
function [x, fval, out] = gtrs_WLK21(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, xi, zeta, opts)
    tic;
    % default parameters
    if ~isfield(opts, 'epsilon'); opts.epsilon = 1e-6; end
    if ~isfield(opts, 'grad_alg'); opts.grad_alg = @grad_descent_WK; end
    if ~isfield(opts, 'termination_criteria'); opts.termination_criteria = @termination_criteria_WK; end
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
    sub = -1;           % sub keeps track of which subprocedure (CRLeft / CRRight / CRMid) to run
    flag = -1;          % flag keeps track of the output case of ConstructReform (regular / maybe regular / not regular)
    gamma = gamma_hat;
    epsilon = opts.epsilon;

    % Step 1 & 2 in ConstructReform (Algorithm 1)
    fprintf('\tComputing nu_0\n');
    mu = xi;
    delta = epsilon / (4 * zeta);
    [~, nu_0] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma);
    out.time_nu0 = toc;

    % Runninng subprocedures (CRLeft / CRRight / CRMid)
    fprintf('\tRunning Subprocedure ');
    opt = struct();
    opt.zeta = zeta;
    if abs(nu_0 - epsilon / (4*zeta)) > 0
        if nu_0 - epsilon / (4*zeta) > 0
            sub = 1;
            fprintf('CRRight\n');
            opt.plus = 1;
        else
            sub = 3;
            fprintf('CRLeft\n');
            opt.plus = 0;
        end

        % CRLeft / CRRight (Algorithm 2)
        maxiter = ceil(log(3200 * zeta^4 / (epsilon * xi^3))); 
        out.time_subs = zeros(1, maxiter);      % time in each iteration of subprocedure
        for t = 1 : maxiter
            tic;
            gamma_prev = gamma;
            mu = mu / 2;        % mu = 2^(-t) * xi
            if t > 1;  opt.v0 = v; end

            % ApproxGammaLeft / ApproxGammaRight (Step 1(b) in Alg.2)
            [gamma, v] = approxGamma(A_0, A_1, gamma_hat, mu, xi, opt);
            % ApproxNu (Step 1(c) in Alg.2)
            [~, nu_t] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma);

            if (sub == 1 && nu_t + epsilon / (4 * zeta) < 0) || (sub == 3 && nu_t - epsilon / (4 * zeta) > 0)
                % "regular" (Step 1(d) in Alg.2)
                flag = 1;
                if sub == 1
                    gamma1 = gamma_prev;
                    gamma2 = gamma;
                else
                    gamma1 = gamma;
                    gamma2 = gamma_prev;
                end
                mu = mu / 2;
                out.time_subs(t) = toc;
                break;
            
            % Step 1(e) in Alg.2
            elseif abs(nu_t) <= epsilon / (4*zeta)
                if sub == 1
                    gamma_p = gamma + mu / 4;
                else
                    gamma_p = gamma - mu / 4;
                end
        
                [~, nu_p] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma_p);
                if (sub == 1 && nu_p + epsilon / (4*zeta) < 0) || (sub == 3 && nu_p - epsilon / (4*zeta) > 0)
                    % "regular" (Step 1(e)iii. in Alg.2)
                    flag = 1;
                    if sub == 1
                        gamma1 = gamma_prev;
                        gamma2 = gamma_p;
                    else
                        gamma1 = gamma_p;
                        gamma2 = gamma;
                    end
                    mu = mu / 4;
                    out.time_subs(t) = toc;
                    break;
                else 
                    % "maybe regular" (Step 1(e)iv. in Alg.2)
                    flag = 2;
                    if sub == 1
                        gamma1 = gamma;
                        gamma2 = gamma_p;
                    else
                        gamma1 = gamma_p;
                        gamma2 = gamma;
                    end
                    mu = mu / 4;
                    out.time_subs(t) = toc;
                    break;
                end
            end
            out.time_subs(t) = toc;
        end
        out.time_subs = out.time_subs(1:t);
        out.time_sub = sum(out.time_subs);
    else 
        % CRMid (Algorithm 3)
        fprintf('CRMid\n');  
        sub = 2;
        tic;

        % Step 2 & 3 in Alg.3    
        gamma1 = gamma_hat - xi / 2;
        gamma2 = gamma_hat + xi / 2;
        [~, nu_1] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma1);
        [~, nu_2] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma2);

        if nu_2 + epsilon / (4 * zeta) < 0 && 0 < nu_1 - epsilon / (4 * zeta)
            % "regular" (Step 4 in Alg.3)
            flag = 1;
            gamma1 = gamma_hat - xi / 2;
            gamma2 = gamma_hat + xi / 2;
            mu = xi / 2;
            out.time_sub = toc;
        elseif nu_1 - epsilon / (4 * zeta) <= 0
            % "maybe regular" (Step 5 in Alg.3)
            flag = 2;
            gamma1 = gamma_hat - xi / 2;
            gamma2 = gamma_hat;
            mu = xi / 2;
            out.time_sub = toc;
        else
            % "maybe regular" (Step 6 in Alg.3)
            flag = 2;
            gamma1 = gamma_hat;
            gamma2 = gamma_hat + xi / 2;
            mu = xi / 2;
            out.time_sub = toc;
        end
    end
    fprintf('\tSubprocedure time: %f\n', out.time_sub);

    % Solve
    fprintf('\tSolving: ');
    if flag == 1
        % SolveRegular (Algorithm 4)
        tic;
        fprintf('SolveRegular\n');
        opts_grad = opts.opts_grad;
        opts_grad.termination_criteria = opts.termination_criteria;
        opts_grad.mu = mu;
        opts_grad.zeta = zeta;
        
        % gradient descent starts with x_gamma
        gamma = 0.5 * (gamma1 + gamma2);
        A_gamma = A_0 + gamma * A_1;
        b_gamma = b_0 + gamma * b_1;
        tol = 1e-12;
        [x0, ~] = grad_descent_CG(A_gamma, -b_gamma, tol);
        opts_grad.x_prev = x0;
        
        if sub == 1
            gamma1 = gamma_hat;
        elseif sub == 3
            gamma2 = gamma_hat;
        end

        fprintf('\t\tgamma1 = %f, gamma2 = %f, mu = %.3E\n', gamma1, gamma2, mu);
        % parameters for gradient descent
        A1 = A_0 + gamma1 * A_1;
        b1 = b_0 + gamma1 * b_1;
        c1 = c_0 + gamma1 * c_1;
        A2 = A_0 + gamma2 * A_1;
        b2 = b_0 + gamma2 * b_1;
        c2 = c_0 + gamma2 * c_1;
        opts_grad.L = max(-eigifp(-A1, opts_L), -eigifp(-A2, opts_L)) + opts_L.tol;
        out.time_L = toc;
        opts_A1 = struct();
        opts_A1.disp = 0;
        opts_A1.tol = 0.1;
        [~, v_plus] = eigifp(A_1, opts_A1);
        [~, v_minus] = eigifp(-A_1, opts_A1);
        opts_grad.v_minus = v_minus;
        opts_grad.v_plus = v_plus;
        
        % gradient descent
        fprintf('\t\tRunning gradient descent\n');
        [x, ~, out_grad] = opts.grad_alg(A1, b1, c1, A2, b2, c2, A_0, b_0, c_0, A_1, b_1, c_1, opts_grad);

        conv = out_grad.conv;
        out.time_log = conv(1, :);
        out.Fcvx = conv(2, :);
        out.Fq0 = conv(3, :);
        out.time_solve = sum(conv(1, :));
        fprintf('\t\tGrad descent time: %f\n', out.time_solve);
        out.fval_cvx = out.Fcvx(end);
        out.success = out_grad.success;
    elseif flag == 2
        % solve "maybe regular"
        tic;
        fprintf('maybe regular\n');
        gamma = (gamma1 + gamma2) / 2;
        A_gamma = A_0 + gamma * A_1;
        b_gamma = b_0 + gamma * b_1;
        tol = mu^2 * epsilon / (20 * zeta);
        [x, ~] = grad_descent_CG(A_gamma, -b_gamma, tol);
        out.time_solve = toc;
    else
        % solve "not regular"
        fprintf('not regular\n');
        A_gamma = A_0 + gamma * A_1;
        b_gamma = b_0 + gamma * b_1;
        mu = mu / 2;
        tol = max(mu^2 * epsilon / (80 * zeta^2), 1e-16);
        opts = struct(); opts.log = 1; opts.A_0 = A_0; opts.b_0 = b_0; opts.c_0 = c_0;
        [x, out_grad] = grad_descent_CG(A_gamma, -b_gamma, tol, opts);
        fprintf('CG: %d, tol = %.3E\n', out_grad.flag, tol);
        conv = out_grad.conv;
        out.time_log = conv(1, :);
        out.Fcvx = conv(2, :);
        out.Fq0 = conv(3, :);
        out.time_solve = sum(conv(1, :));
        out.time_L = 0;
        opts_A1 = struct();
        opts_A1.disp = 0;
        opts_A1.tol = 1e-5;
        [~, v_plus] = eigifp(A_1, opts_A1);
        [~, v_minus] = eigifp(-A_1, opts_A1);
    end

    % Rounding (detailed in Section 4.1)
    tic;
    if sub ~= 2
        fprintf('\tRounding\n');
        x = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, v_minus, v_plus);
    end
    out.time_rounding = toc;

    fval = quad_eval(A_0, b_0, c_0, x);
    out.x = x;
    out.fval = fval;
    out.q1 = quad_eval(A_1, b_1, c_1, x);
    out.itr = out_grad.itr;
    out.time = out.time_nu0 + out.time_sub + out.time_L + out.time_solve + out.time_rounding;
end
