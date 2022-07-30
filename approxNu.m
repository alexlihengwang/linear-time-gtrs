% approxNu.m
% ApproxNu (Algorithm 6)
function [x, nu] = approxNu(A_0, b_0, c_0, A_1, b_1, c_1, zeta, mu, delta, gamma)
    A_gamma = A_0 + gamma * A_1;
    b_gamma = b_0 + gamma * b_1;

    % tol in grad_descent_CG is computed as ||Ax + b||.
    tol = mu * (mu * delta / (10 * zeta));      % Given ||(Ax_1 + b) - (Ax_2 + b)|| <= ||A||_2 * ||x_1 - x_2|| and ||A||_2 >= mu  
    [x, ~] = grad_descent_CG(A_gamma, -b_gamma, tol);
    nu = quad_eval(A_1, b_1, c_1, x);
end
