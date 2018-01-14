function predMat = predAdjMatDsbm(adj,classProbMat,class)
%predAdjMatDsbm Forecast adjacency matrix at next time step using DSBM
%   predMat = predAdjMatDsbm(adj,edgeProbMat,class)
%
%   Inputs:
%   adj - 3-D array of graph adjacency matrices, where each slice along the
%         third dimension denotes the adjacency matrix at time t. Each
%         adjacency matrix is binary with no self-edges and can be directed,
%         i.e. w(i,j,t) = 1 denotes an edge from i to j at time t, and
%         w(i,j,t) = 0 denotes the absence of an edge from i to j at time t.
%   classProbMat - 3-D array of class connection probability matrices,
%                  where entry (a,b,t) denotes the probability of a node in
%                  class a at time t connecting to a node in class b at
%                  time t
%   class - Matrix of class membership vectors, where each column denotes
%           the class membership at time t. Set elements to 0 to indicate
%           that a node is inactive during a time step.
%
%   Outputs:
%   predMat - 3-D array of edge probabilities, where entry (i,j,t) denotes
%             the probability of forming an edge from node i to j at time t

% Author: Kevin S. Xu

[n,~,tMax] = size(adj);
k = size(classProbMat,1);

predMat = zeros(n,n,tMax);
for t = 2:tMax
    predMatCurr = zeros(n,n);
    for c1 = 1:k
        for c2 = 1:k
            predMatCurr(class(:,t-1)==c1,class(:,t-1)==c2) ...
                = classProbMat(c1,c2,t-1);
        end
    end
    predMatCurr(diag(true(n,1))) = 0;    % Zero out diagonal
    predMat(:,:,t) = predMatCurr;
end

end

