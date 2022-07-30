% termination_criteria_WK.m
% termination criteria of the gradient descent algorithms in WK20 and WLK21.
function flag = termination_criteria_WK(A_0, b_0, c_0, A_1, b_1, c_1, L, x_new, y_old, x_old, epsilon)
    flag = false;
    gen_gradient = 2 * L * (y_old - x_new);
    if norm(gen_gradient) <= epsilon 
        flag = true;
    end
end
