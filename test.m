% Demo script to run algorithms(WLK21, WK20, JL19, AN19, BTH14) for the GTRS 
% Requirment: MOSEK installed and added to PATH


% parameters
n = 1e3;
density = 1e-2;
mu_star = 1e-2;
xi = 0.1;

% GTRS solvers to run
run_AN19 = true;
run_BTH14 = true;
run_JL19 = true;


% random feasible instances
tic
[A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, zeta, opt] = definite_feasible_instance(n, xi, density, mu_star);
toc
fprintf('computed instance\n\n');
            

% WLK21
fprintf('Running WLK21\n');
opts = struct();
[~, ~, out_WLK21] = gtrs_WLK21(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, xi, zeta, opts);
time_WLK21 = out_WLK21.time;
err_WLK21 = out_WLK21.fval - opt;
fprintf('WLK21 Error: %.3E, Time: %.3E\n\n', abs(err_WLK21), out_WLK21.time);
            
% WK20
fprintf('Running WK20\n');
opts = struct();
[~, ~, out_WK20] = gtrs_WK20(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
time_WK20 = out_WK20.time;
err_WK20 = out_WK20.fval - opt;
fprintf('WK20 Error: %.3E, Time: %.3E\n\n', err_WK20, time_WK20);

% JL19
if run_JL19
fprintf('Running JL19\n');
opts = struct();
opts.grad_alg = @grad_descent_JL;
opts.termination_criteria = @termination_criteria_WK;
[~, ~, out_JL19] = gtrs_WK20(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
time_JL19 = out_JL19.time;
err_JL19 = out_JL19.fval - opt;
fprintf('JL19 Error: %.3E, Time: %.3E\n\n', err_JL19, time_JL19);
end
            
% AN19
if run_AN19
fprintf('Running AN19\n');
opts = struct();
[~, ~, out_AN19] = gtrs_AN19(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
time_AN19 = out_AN19.time;
err_AN19 = abs(out_AN19.fval - opt);
fprintf('AN19 Error: %.3E, Time: %.3E\n\n', err_AN19, time_AN19);
end
            
% BTH14
if run_BTH14
fprintf('Running BTH14\n');
[~, ~, out_BTH14] = gtrs_BTH14(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, './mosek_log.txt');
time_BTH14 = out_BTH14.time;
err_BTH14 = abs(out_BTH14.fval - opt);
fprintf('BTH14 Error: %.3E, Time: %.3E\n\n', err_BTH14, time_BTH14);
end

