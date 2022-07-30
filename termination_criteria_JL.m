% termination_criteria_JL.m
% termination criteria of the gradient descent algorithm in JL19.
function flag = termination_criteria_JL(A_0, b_0, c_0, A_1, b_1, c_1, L, x_new, y_old, x_old)
    epsilon1 = 1e-10;
    epsilon2 = 1e-13;
    epsilon3 = 1e-10;
    
    flag = false;

    f0_old = (x_old' * A_0 * x_old + 2 * b_0' * x_old + c_0);
    f1_old = (x_old' * A_1 * x_old + 2 * b_1' * x_old + c_1);
    f0_new = (x_new' * A_0 * x_new + 2 * b_0' * x_new + c_0);
    f1_new = (x_new' * A_1 * x_new + 2 * b_1' * x_new + c_1);
    F_old = max(f0_old, f1_old);
    F_new = max(f0_new, f1_new);

    g0 = 2 * (A_0 * x_new + b_0);
    g1 = 2 * (A_1 * x_new + b_1);

    if F_old - F_new < epsilon2
        flag = true;
        return;
    end

    if (abs(f0_new - f1_new)/(abs(f0_new) + abs(f1_new))) < epsilon1
        alpha = - dot(g0, g1 - g0) / dot(g1 - g0, g1 - g0);
        alpha = max(alpha, 0);
        alpha = min(alpha, 1);
        if norm((1-alpha) * g0 + alpha * g1) < epsilon3
            flag = true;
            return;
        end
    end

    if (abs(f0_new - f1_new)/(abs(f0_new) + abs(f1_new))) > epsilon1
        if f0_new > f1_new
            g = g0;
        else
            g = g1;
        end
        
        if norm(g) < epsilon3
            flag = true;
            return
        end
    end
end
