function [adj,probMarg,scaleFactors] = generateSbtm(class,probInit, ...
    probNew,probExist,directed)
%generateSbtm Generate realization of stochastic block transition model
%   [adj,probMarg,scaleFactors] = generateSbtm(class,probInit, ...
%       probNew,probExist,directed)
%
%   generateSbtm generates a sequence of graph adjacency matrices using a
%   stochastic block transition model (SBTM).
%
%   Inputs:
%   class - Matrix of class memberships where each column denotes the class
%           memberships at time t. Set entries to 0 to denote that a node was
%           not present at a particular time.
%   probInit - Matrix of edge probabilities between blocks at initial time
%              step
%   probNew - 3-D array where each slice denotes the matrix of probabilities
%             of forming new edges in each block. The initial slice
%             probNew(:,:,1) is ignored.
%   probExist - 3-D array where each slice denotes the matrix of
%               probabilities of previous edges re-occurring in each
%               block. The initial slice probExist(:,:,1) is ignored.
%   directed - Whether the graph is directed (set to true or false,
%              defaults to false)
%
%   Outputs:
%   adj - 3-D array of adjacency matrices, where the t-th slice denotes the
%         adjacency matrix generated at time t
%   probMarg - 3-D array where the t-th slice is a matrix of marginal
%              probabilities of edges within blocks at time t
%   scaleFactors - 3-D array where the t-th slice denotes the matrix of
%                  ground-truth scaling factors for the SBTM at time t

% Author: Kevin S. Xu

% Set as undirected graph by default
if nargin < 5
	directed = false;
end

[n,tMax] = size(class);
k = max(class(:));  % Maximum number of classes over all time steps
adj = zeros(n,n,tMax);

% Generate initial snapshot according to an SBM
adj(:,:,1) = generateSbm(class(:,1),probInit,directed);
% Marginal probabilities of forming edges between blocks at each time
probMarg = zeros(k,k,tMax);
probMarg(:,:,1) = probInit;
scaleFactors = ones(n,n,tMax);

% Generate subsequent snapshots using SBTM
for t = 2:tMax
    % Update marginal probabilities of forming edges between blocks
    probMarg(:,:,t) = probNew(:,:,t).*(1-probMarg(:,:,t-1)) ...
        + probExist(:,:,t).*probMarg(:,:,t-1);
    
    % Calculate scaling factors for edge probabilities
    scaleFactors(:,:,t) = calcSbtmScaleFactors(adj(:,:,t-1),class(:,t), ...
        class(:,t-1),probNew(:,:,t),probExist(:,:,t),probMarg(:,:,t-1), ...
        directed);
    
    for i = 1:n
        % Skip this node if it was not present at time t
        if class(i,t) == 0
            continue
        end
        
        % Loop start index depends on whether graph is directed or
        % undirected
        if directed == true
            start = 1;
        else
            start = i+1;
        end
        
        for j = start:n
            % Ignore self-edges
            if i == j
                continue
            end
            
            % Skip this node if it was not present at time t
            if class(j,t) == 0
                continue
            end
            
            if adj(i,j,t-1) == 0
                % No edge between and i and j previously so the edge
                % probability is a scaled version of probNew
                edgeProb = scaleFactors(i,j,t)*probNew(class(i,t), ...
                    class(j,t),t);
            else
                % Edge between i and j exists previously so the edge
                % probability is a scaled version of probExist
                edgeProb = scaleFactors(i,j,t)*probExist(class(i,t), ...
                    class(j,t),t);
            end
            
%             assert((edgeProb>=0) && (edgeProb<=1),['Edge probability ' ...
%                 'must be between 0 and 1. Currently ' num2str(edgeProb)]);
            adj(i,j,t) = bernrnd(edgeProb);
            if directed == false
                adj(j,i,t) = adj(i,j,t);
            end
        end
    end
end

end
