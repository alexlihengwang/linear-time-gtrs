% quad_eval.m
% Return the value of quadratic function q(x) = x' * A * x + 2 * b' * x + c.
function output = quad_eval(A, b, c, x)
    output = x' * A * x + 2 * b' * x + c;
end
