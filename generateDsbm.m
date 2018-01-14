function W = generateDsbm(C,P,directed)
%generateDsbm Generate realization of dynamic stochastic block model
%   generateDsbm(C,P) generates a sequence of adjacency matrices
%   where each adjacency matrix is a realization of a stochastic block
%   model with class memberships given by a column of C and block
%   probabilities given by a slice (third dimension) of P. Nodes that are
%   not present at a particular time are denoted by zeros in the
%   corresponding column of C.
%
%   generateDsbm(C,P,directed) allows for creation of direted graphs
%   by setting directed to true. By deafult, undirected graphs are created.

% Author: Kevin S. Xu

% Set as undirected graph by default
if nargin == 2
    directed = false;
end

[n,tMax] = size(C);
W = zeros(n,n,tMax);
% Generate realization of stochastic block model at each time step
for t = 1:tMax
    W(:,:,t) = generateSbm(C(:,t),P(:,:,t),directed);
end

end

