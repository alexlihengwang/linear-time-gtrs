% Numerical tests reported in the paper
% run this script to generate plots and tables
% Requirment: MOSEK installed and added to PATH


clear;

% dir to save outputs
dir = 'outputs_20230118';
mkdir(dir);

% dimension 
n = 1e3;
dir_name = strcat(dir, '/results_server_1e', string(log10(n)));
mkdir(dir_name);

% density and regularity
densities = [100 / n, 10 / n];
mus = [1e-2, 1e-4, 1e-6];
xi = 0.1;

if n == 1e5
    N = 5;      % number of tests to run 
else
    N = 100;
end

% GTRS solvers to run
run_AN19 = (n <= 1e4);
run_BTH14 = (n <= 1e3);
run_JL19 = true;

% initialize arrays to save output
Time_WLK21 = zeros(1,N); Time_WLK21_eig = zeros(1,N); Time_WLK21_grad = zeros(1,N); Err_WLK21 = zeros(1,N); Err_WLK21_cvx = zeros(1,N);
Time_WK20 = zeros(1,N); Time_WK20_eig = zeros(1,N); Time_WK20_grad = zeros(1,N); Err_WK20 = zeros(1,N); Err_WK20_cvx = zeros(1,N);
if run_AN19; Time_AN19 = zeros(1,N); Err_AN19 = zeros(1,N); end
if run_BTH14; Time_BTH14 = zeros(1,N); Err_BTH14 = zeros(1,N); end
if run_JL19; Time_JL19 = zeros(1,N); Time_JL19_eig = zeros(1,N); Time_JL19_grad = zeros(1,N); Err_JL19 = zeros(1,N); Err_JL19_cvx = zeros(1,N); end

fig = figure();

% test instances with given densities and mu_star
for i = 1 : length(densities)
    density = densities(i);
    for j = 1 : length(mus)
        mu_opt = mus(j);
        maxlen = 0;
        % test on N random instances
        D_WLK21 = []; D_WK20 = []; D_JL19 = []; D_AN19 = []; D_BTH14 = []; D_WLK21_cvx = []; D_WK20_cvx = []; D_JL19_cvx = [];
        for mm = 1 : N
            fprintf('Instance %d out of %d\n', (i - 1) * length(mus) * N + (j - 1) * N + mm, length(densities) * length(mus) * N);
            tic
            [A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, zeta, opt] = definite_feasible_instance(n, xi, density, mu_opt);
            toc
            fprintf('computed instance\n\n');
            
            % WLK21
            fprintf('Running WLK21\n');
            opts = struct();
            [~, ~, out_WLK21] = gtrs_WLK21(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, xi, zeta, opts);
            L = length(out_WLK21.time_log);
            for k = 1:L
                out_WLK21.time_log(L-k+1) = sum(out_WLK21.time_log(1:(L-k+1)));
            end
            time_WLK21 = out_WLK21.time_log + out_WLK21.time_nu0 + out_WLK21.time_sub + out_WLK21.time_L;
            err_WLK21_cvx = out_WLK21.Fcvx - opt;
            err_WLK21_q0 = out_WLK21.Fq0 - opt;
            fprintf('WLK21 Error: %.3E, Time: %.3E\n\n', abs(out_WLK21.fval-opt), out_WLK21.time);
            
            % WK20
            fprintf('Running WK20\n');
            opts = struct();
            [~, ~, out_WK20] = gtrs_WK20(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
            L = length(out_WK20.time_log);
            for k = 1:L
                out_WK20.time_log(L-k+1) = sum(out_WK20.time_log(1:(L-k+1)));
            end
            time_WK20 = out_WK20.time_log + out_WK20.time_eig + out_WK20.time_L;
            err_WK20_cvx = out_WK20.Fcvx - opt;
            err_WK20_q0 = out_WK20.Fq0 - opt;
            fprintf('WK20 Error: %.3E, Time: %.3E\n\n', abs(out_WK20.fval-opt), out_WK20.time);
            
            % AN19
            if run_AN19
            fprintf('Running AN19\n');
            opts = struct();
            opts.log = 1;
            [~, ~, out_AN19] = gtrs_AN19(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
            time_AN19 = out_AN19.time_log;
            err_AN19 = abs(out_AN19.Fval - opt);
            fprintf('AN19 Error: %.3E, Time: %.3E\n\n', abs(out_AN19.fval-opt), out_AN19.time);
            end
            
            % BTH14
            if run_BTH14
                fprintf('Running BTH14\n');
                [~, ~, out_BTH14] = gtrs_BTH14(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat);
                fprintf('BTH14 Error: %.3E, Time: %.3E\n\n', abs(out_BTH14.fval-opt), out_BTH14.time);
                
                % retrieve convergence data from mosek log
                fileID = fopen('results/mosek_log.txt');
                while true
                    c = textscan(fileID, '%s', 1);
                    if strcmp(c{1}{1}, 'TIME')
                        break;
                    end
                end
                data = textscan(fileID, '%f', 'Delimiter', ',', 'MultipleDelimsAsOne', 1);
                fclose(fileID);
                data = data{1};
                len = length(data);
                time_BTH14 = []; err_BTH14 = [];
                for k = 1:len
                    if mod(k,9) == 6
                        err = abs(opt + data(k));
                        err_BTH14 = [err_BTH14, err];
                    elseif mod(k,9) == 0
                        time_BTH14 = [time_BTH14, data(k) + out_BTH14.time - out_BTH14.t_mosek];
                    end
                end
            end

            % JL19
            if run_JL19
            fprintf('Running JL19\n');
            opts = struct();
            opts.grad_alg = @grad_descent_JL;
            opts.termination_criteria = @termination_criteria_WK;
            if n == 1e5 && density == 1e-3
                opts_grad = struct();
                opts_grad.maxit = 10000;
                opts.opts_grad = opts_grad;
            end
            
            [~, ~, out_JL19] = gtrs_WK20(A_0, b_0, c_0, A_1, b_1, c_1, gamma_hat, opts);
            fprintf('JL19 Error: %.3E, Time: %.3E\n\n', abs(out_JL19.fval-opt), out_JL19.time);
            
            L = length(out_JL19.time_log);
            for k = 1:L
                out_JL19.time_log(L-k+1) = sum(out_JL19.time_log(1:(L-k+1)));
            end
            time_JL19 = out_JL19.time_log + out_JL19.time_eig + out_JL19.time_L;
            err_JL19_cvx = out_JL19.Fcvx - opt; 
            err_JL19_q0 = out_JL19.Fq0 - opt;
            end


            % Plot convergence
            subplot(length(densities), length(mus), (i-1) * length(mus) + j);
            semilogy(time_WLK21, err_WLK21_q0, 'r');
            hold on
            semilogy(time_WK20, err_WK20_q0, 'b');
            if run_AN19; semilogy(time_AN19, err_AN19, 'Color', [1, 0, 1]); end
            if run_BTH14; semilogy(time_BTH14, err_BTH14, 'g'); end
            if run_JL19; semilogy(time_JL19, err_JL19_q0, 'black'); end
            
            % Save iterates
            l = size(D_WLK21);
            if length(time_WLK21) < l(2)
                D_WLK21 = [D_WLK21; time_WLK21, NaN(1, l(2)-length(time_WLK21)); err_WLK21_q0, NaN(1, l(2)-length(time_WLK21))];
            else
                D_WLK21 = [D_WLK21, NaN(l(1), length(time_WLK21)-l(2)); time_WLK21; err_WLK21_q0];
            end

            l = size(D_WLK21_cvx);
            if length(time_WLK21) < l(2)
                D_WLK21_cvx = [D_WLK21_cvx; time_WLK21, NaN(1, l(2)-length(time_WLK21)); err_WLK21_cvx, NaN(1, l(2)-length(time_WLK21))];
            else
                D_WLK21_cvx = [D_WLK21_cvx, NaN(l(1), length(time_WLK21)-l(2)); time_WLK21; err_WLK21_cvx];
            end
            
            l = size(D_WK20);
            if length(time_WK20) < l(2)
                D_WK20 = [D_WK20; time_WK20, NaN(1, l(2)-length(time_WK20)); err_WK20_q0, NaN(1, l(2)-length(time_WK20))];
            else
                D_WK20 = [D_WK20, NaN(l(1), length(time_WK20)-l(2)); time_WK20; err_WK20_q0];
            end

            l = size(D_WK20_cvx);
            if length(time_WK20) < l(2)
                D_WK20_cvx = [D_WK20_cvx; time_WK20, NaN(1, l(2)-length(time_WK20)); err_WK20_cvx, NaN(1, l(2)-length(time_WK20))];
            else
                D_WK20_cvx = [D_WK20_cvx, NaN(l(1), length(time_WK20)-l(2)); time_WK20; err_WK20_cvx];
            end
            
            if run_AN19
                l = size(D_AN19);
                if length(time_AN19) < l(2)
                    D_AN19 = [D_AN19; time_AN19, NaN(1, l(2)-length(time_AN19)); err_AN19, NaN(1, l(2)-length(time_AN19))];
                else
                    D_AN19 = [D_AN19, NaN(l(1), length(time_AN19)-l(2)); time_AN19; err_AN19];
                end
            end
            
            if run_BTH14
                l = size(D_BTH14);
                if length(time_BTH14) < l(2)
                    D_BTH14 = [D_BTH14; time_BTH14, NaN(1, l(2)-length(time_BTH14)); err_BTH14, NaN(1, l(2)-length(time_BTH14))];
                else
                    D_BTH14 = [D_BTH14, NaN(l(1), length(time_BTH14)-l(2)); time_BTH14; err_BTH14];
                end
            end
            
            if run_JL19
                l = size(D_JL19);
                if length(time_JL19) < l(2)
                    D_JL19 = [D_JL19; time_JL19, NaN(1, l(2)-length(time_JL19)); err_JL19_q0, NaN(1, l(2)-length(err_JL19_q0))];
                else
                    D_JL19 = [D_JL19, NaN(l(1), length(time_JL19)-l(2)); time_JL19; err_JL19_q0];
                end
            end   

            if run_JL19
                l = size(D_JL19_cvx);
                if length(time_JL19) < l(2)
                    D_JL19_cvx = [D_JL19_cvx; time_JL19, NaN(1, l(2)-length(time_JL19)); err_JL19_cvx, NaN(1, l(2)-length(err_JL19_cvx))];
                else
                    D_JL19_cvx = [D_JL19_cvx, NaN(l(1), length(time_JL19)-l(2)); time_JL19; err_JL19_cvx];
                end
            end
                           
            % Save to tables
            Time_WLK21(mm) = out_WLK21.time;
            Time_WLK21_eig(mm) = out_WLK21.time_sub;
            Time_WLK21_grad(mm) = out_WLK21.time_solve;
            Err_WLK21(mm) = abs(out_WLK21.fval - opt);
            Err_WLK21_cvx(mm) = abs(out_WLK21.fval_cvx - opt);
            if length(out_WLK21.time_subs) > maxlen
                maxlen = length(out_WLK21.time_subs);
            end
                 
            Time_WK20(mm) = out_WK20.time;
            Time_WK20_eig(mm) = out_WK20.time_eig;
            Time_WK20_grad(mm) = out_WK20.time_grad;
            Err_WK20(mm) = abs(out_WK20.fval - opt);
            Err_WK20_cvx(mm) = abs(out_WK20.fval_cvx - opt);
            
            if run_JL19
                Time_JL19(mm) = out_JL19.time; 
                Time_JL19_eig(mm) = out_JL19.time_eig;
                Time_JL19_grad(mm) = out_JL19.time_grad;
                Err_JL19(mm) = abs(out_JL19.fval - opt);
                Err_JL19_cvx(mm) = abs(out_JL19.fval_cvx - opt);
            end

            if run_AN19; Time_AN19(mm) = out_AN19.time; Err_AN19(mm) = abs(out_AN19.fval - opt); end
            if run_BTH14; Time_BTH14(mm) = out_BTH14.time; Err_BTH14(mm) = abs(out_BTH14.fval - opt); end
        end
        
        avg_time = [mean(Time_WLK21); mean(Time_WK20); mean(Time_WLK21); mean(Time_WK20)];
        avg_err = [mean(Err_WLK21); mean(Err_WK20); mean(Err_WLK21_cvx); mean(Err_WK20_cvx)];
        RowNames = {'WLK21'; 'WK20'; 'WLK21_CR'; 'WK20_CR'};
        if run_AN19
            avg_time = [avg_time; mean(Time_AN19)];
            avg_err = [avg_err; mean(Err_AN19)];
            RowNames = [RowNames; 'AN19'];
        end
        if run_BTH14   
            avg_time = [avg_time; mean(Time_BTH14)];
            avg_err = [avg_err; mean(Err_BTH14)];        
            RowNames = [RowNames; 'BTH14'];
        end
        if run_JL19
            avg_time = [avg_time; mean(Time_JL19); mean(Time_JL19)];
            avg_err = [avg_err; mean(Err_JL19); mean(Err_JL19_cvx)];        
            RowNames = [RowNames; 'JL19'; 'JL19_CR'];
        end
        T_avg = table(avg_time, avg_err, 'RowNames', RowNames);
        
        log = [Time_WLK21; Err_WLK21; Err_WLK21_cvx; Time_WLK21_eig; Time_WLK21_grad; Time_WK20; Err_WK20; Err_WK20_cvx; Time_WK20_eig; Time_WK20_grad];
        RowNames = {'Time_WLK21'; 'Error_WLK21'; 'Err_WLK21_CR'; 'Time_WLK21_eig'; 'Time_WLK21_grad'; 'Time_WK20'; 'Error_WK20'; 'Err_WK20_CR'; 'Time_WK20_eig'; 'Time_WK20_grad'};
        if run_JL19
            log = [log; Time_JL19; Err_JL19; Err_JL19_cvx; Time_JL19_eig; Time_JL19_grad];
            RowNames = [RowNames; 'Time_JL19'; 'Error_JL19'; 'Error_JL19_CR'; 'Time_JL19_eig'; 'Time_JL19_grad'];
        end         
        if run_BTH14
            log = [log; Time_BTH14; Err_BTH14];
            RowNames = [RowNames; 'Time_BTH14'; 'Error_BTH14'];
        end
        if run_AN19
            log = [log; Time_AN19; Err_AN19];
            RowNames = [RowNames; 'Time_AN19'; 'Error_AN19'];
        end
        T_log = table(log, 'RowNames', RowNames);

        iterations_data = [out_WLK21.itr; out_WK20.itr_grad];
        RowNames = {'iterations_WLK21'; 'iterations_WK20'};
        T_itr = table(iterations_data, 'RowNames', RowNames);
        
        avg_time_eig = [mean(Time_WLK21_eig); mean(Time_WK20_eig)];
        avg_time_grad = [mean(Time_WLK21_grad); mean(Time_WK20_grad)];
        RowNames = {'WLK21'; 'WK20'};
        if run_JL19
            avg_time_eig = [avg_time_eig; mean(Time_JL19_eig)];
            avg_time_grad = [avg_time_grad; mean(Time_JL19_grad)];
            RowNames = [RowNames; 'JL19'];
        end
        T_avg_eig = table(avg_time_eig, avg_time_grad, 'RowNames', RowNames);
               
        writetable(T_avg_eig, strcat(dir_name,'/avg_eig_density=',string(density),...
            '_mu=', string(mu_opt), '_n=', string(n), '.csv'), 'WriteRowNames', true);
        writetable(T_avg, strcat(dir_name,'/avg_density=',string(density),...
            '_mu=', string(mu_opt), '_n=', string(n), '.csv'), 'WriteRowNames', true);
        writetable(T_log, strcat(dir_name,'/log_density=',string(density),...
            '_mu=', string(mu_opt), '_n=', string(n), '.csv'), 'WriteRowNames', true);
        writetable(T_itr, strcat(dir_name,'/iterations_density=',string(density),...
            '_mu=', string(mu_opt), '_n=', string(n), '.csv'), 'WriteRowNames', true);
        
        % iterates for plots
        writetable(table(D_WLK21), strcat(dir_name, '/iterates_WLK21_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_WK20), strcat(dir_name, '/iterates_WK20_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_WLK21_cvx), strcat(dir_name, '/iterates_WLK21(ErrorCR)_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_WK20_cvx), strcat(dir_name, '/iterates_WK20(ErrorCR)_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        
        writetable(table(D_AN19), strcat(dir_name, '/iterates_AN19_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_BTH14), strcat(dir_name, '/iterates_BTH14_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_JL19), strcat(dir_name, '/iterates_JL19_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        writetable(table(D_JL19_cvx), strcat(dir_name, '/iterates_JL19(ErrorCR)_N=', string(density*n),...
            '_mu=', compose("%.0e", mu_opt), '_n=', string(n), '.csv'), 'WriteVariableNames', 0);
        
        hold off
        grid on
        title(strcat('\mu* =', string(mu_opt), ', density=', string(density)));
        xlabel('Time (s)');
        if j == 1
            ylabel('Error');
        end
        
        ylim([1e-14, 1e5]);
        yticks([1e-12, 1e-8, 1e-4, 1e0, 1e4])
    end
end

savefig(strcat(dir_name,'/convergence_grid_n=', string(n), '.fig'))

