function x = bernrnd(p,m,n)
% bernrnd(p,m,n) generates an m-by-n matrix of realizations of Bernoulli 
% random variables with parameter p.

if nargin < 2
    m = 1;
end
if nargin < 3
    n = m;
end

assert((p>=0) && (p<=1), ['p must be between 0 and 1. Currently p = ' ...
    num2str(p)])
x = binornd(1,p,m,n);

% % Implementation not requiring Statistics Toolbox
% y = rand(m,n);	% Uniformly distributed random variables
% x = zeros(m,n);
% x(y < p) = 1;
