% rounding.m
% Rounding step used in WK20, WLK21 and JL19 as detailed in Section 4.1
function [x] = rounding(x, A_0, b_0, c_0, A_1, b_1, c_1, v_minus, v_plus)
    q_1 = quad_eval(A_1, b_1, c_1, x);
    if q_1 ~= 0
        if q_1 > 0
            v = v_plus;
        else
            v = v_minus;
        end
        [alpha_minus, alpha_plus] = solve_quad(v' * A_1 * v, v' * A_1 * x + v' * b_1, q_1);
        q_minus = quad_eval(A_0, b_0, c_0, x + alpha_minus * v);
        q_plus = quad_eval(A_0, b_0, c_0, x + alpha_plus * v);
        if q_minus < q_plus
            x = x + alpha_minus * v;
        else
            x = x + alpha_plus * v;
        end
    end
end


function [sol1, sol2] = solve_quad(a, b, c)
    % assume that the discriminant is nonnegative
    b_norm = b / a;
    c_norm = c / a;
    if b_norm^2 - c_norm < 0
        warning('Negaive in sqrt()') 
    end
    disc = sqrt(b_norm^2 - c_norm);
    sol1 = - b_norm - disc;
    sol2 = - b_norm + disc;
end
