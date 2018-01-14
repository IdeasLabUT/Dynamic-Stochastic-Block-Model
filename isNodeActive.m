function nodeActive = isNodeActive(adj)
%isNodeActive Identify active nodes at each time step
%   isNodeActive(adj) returns an n x tMax matrix where entry (i,t) is true
%   if node i is active at time t. A node i is considered active at time t
%   if there is at least one edge to or from i at time t.

% Author: Kevin S. Xu

[n,~,tMax] = size(adj);

% Identify active nodes (nodes with at least one edge to or from the node)
% at each time step
nodeActive = false(n,tMax);
parfor t = 1:tMax
    nodeActive(:,t) = (sum(adj(:,:,t)) > 0) | (sum(adj(:,:,t),2) > 0)';
end

end

