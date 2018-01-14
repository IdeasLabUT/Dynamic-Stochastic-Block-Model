function [newDens,existDens,nNewEdges,nExistEdges,nNewPairs, ...
    nExistPairs,Opt] = calcNewExistDens(adj,class,scaleFactors,Opt)
%calcNewExistDens Calculate densities of new and existing edges in blocks
%   [newDens,existDens,nNewEdges,nExistEdges,nNewPairs, ...
%       nExistPairs,Opt] = calcNewExistDens(adj,class,scaleFactors,Opt)
%
%   calcNewExistDens computes the scaled densities of new and existing
%   edges in blocks in the sequence of adjacency matrices adj, where the
%   third dimension indexes time. Blocks are defined by the class
%   membership matrix class, where the i-th column of class denotes the
%   memberships for the i-th slice of adj.
%
%   Inputs:
%   adj - n x n x tMax array where each slice denotes the adjacency
%         matrix of the graph snapshot at a given time step, n denotes the
%         number of nodes, and tMax denotes the number of time steps.
%   class - n x tMax matrix where entry (i,t) denotes the class membership
%           of node i at time t (scalar from 1 to the number of classes k).
%   scaleFactors - n x n x tMax array of scaling factors to use when
%                  computing scaled number of edges and scaled block
%                  densities. Each slice denotes the scaling factors at a
%                  given time step.
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%   'emptyVal' - Density value to set for empty blocks, which occur when a
%                block has no new pairs or no re-occurring pairs [ 1e-3 ]
%   'obsClip' - How much to clip the range of the estimates of edge
%               probabilities away from the boundaries of 0 and 1, for
%               which the logit is undefined. All entries smaller than
%               obsClip and larger than 1-obsClip are clipped. [ 1e-3 ]
%   'nClasses' - Number of classes [ max(class(:)) ]
%
%   Outputs:
%   newDens - p x tMax matrix of scaled densities of new edges in blocks,
%             where p = k*(k+1)/2 for undirected graphs and p = k^2 for
%             directed graphs. Each column corresponds to a vector
%             representation of new block densities from each slice of adj.
%   existDens - p x tMax matrix of scaled densities of re-occuring edges
%               in blocks
%   nNewEdges - p x tMax matrix containing the scaled number of new edges
%               in blocks
%   nExistEdges - p x tMax matrix containing the scaled number of
%                 re-occuring edges in blocks
%   nNewPairs - p x tMax matrix containing the number of new pairs
%               (possible new edges) in blocks
%   nExistPairs - p x tMax matrix containing the number of re-occuring
%                 pairs (possible re-occurring edges) in blocks
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed','emptyVal','obsClip','nClasses'};
defaultValues = {false,1e-3,1e-3,[]};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
emptyVal = Opt.emptyVal;
obsClip = Opt.obsClip;
nClasses = Opt.nClasses;

[n,~,tMax] = size(adj);  % Number of nodes and snapshots (time steps)
% Maximum number of classes over all time steps
if isempty(nClasses)
    kMax = max(class(:));
else
    kMax = nClasses;
end
% Length of density vector depends on whether graphs are directed or
% undirected
if directed == true
    pMax = kMax^2;
else
    pMax = kMax*(kMax+1)/2;
end
% If class membership is specified by a vector, repeat it into a matrix
% with a column vector for each time
if isvector(class)
    class = repmat(reshape(class,n,1),1,tMax);
end

% Scaled number of new and existing edges in each block
nNewEdges = zeros(pMax,tMax);
nExistEdges = zeros(pMax,tMax);
% Number of pairs of new and existing nodes (number of possible new and
% existing edges, respectively) in each block
nNewPairs = zeros(pMax,tMax);
nExistPairs = zeros(pMax,tMax);
% Scaled density (scaled number of edges divided by possible edges) of new
% and existing edges in each each block
newDens = zeros(pMax,tMax);
existDens = zeros(pMax,tMax);

% Identify active nodes at each time step
nodeActive = isNodeActive(adj);

for t = 2:tMax
    % Limit entries corresponding to active nodes at time t
    adjCurr = adj(nodeActive(:,t),nodeActive(:,t),t);
    adjPrev = adj(nodeActive(:,t),nodeActive(:,t),t-1);
    cCurr = class(nodeActive(:,t),t);
    scaleCurr = scaleFactors(nodeActive(:,t),nodeActive(:,t),t);
    nCurr = sum(nodeActive(:,t));
	k = kMax;
%     k = max(cCurr);
    % Scaled adjacency matrix at current time step
    adjScaledCurr = adjCurr./scaleCurr;

    % Binary masks denoting edges and non-edges at previous time
    zeroMask = (adjPrev==0);
    oneMask = ~zeroMask;
    % Remove diagonal from set of non-edges since we are not allowing
    % self-edges
    zeroMask(diag(true(nCurr,1))) = false;

    % Scaled number of new edges
    nNewEdgesMat = zeros(k,k);
    % Scaled number of re-occurring edges
    nExistEdgesMat = zeros(k,k);
    % Number of new pairs (possible new edges)
    nNewPairsMat = zeros(k,k);
    % Number of re-occurring pairs (possible re-occurring edges)
    nExistPairsMat = zeros(k,k);
    % Calculate scaled number of new and existing edges and number of possible
    % new and existing edges in each block
    for c1 = 1:k
        for c2 = 1:k
            blockMask = false(nCurr,nCurr);
            blockMask(cCurr==c1,cCurr==c2) = true;
            zeroBlock = zeroMask & blockMask;
            oneBlock = oneMask & blockMask;

            nNewEdgesMat(c1,c2) = sum(adjScaledCurr(zeroBlock));
            nExistEdgesMat(c1,c2) = sum(adjScaledCurr(oneBlock));
            nNewPairsMat(c1,c2) = sum(zeroBlock(:));
            nExistPairsMat(c1,c2) = sum(oneBlock(:));        
        end
    end

    % For undirected graphs, halve the number edges and pairs along diagonal
    % blocks
    if directed == false
        nNewEdgesMat(diag(true(k,1))) = nNewEdgesMat(diag(true(k,1)))/2;
        nExistEdgesMat(diag(true(k,1))) = nExistEdgesMat(diag(true(k,1)))/2;
        nNewPairsMat(diag(true(k,1))) = nNewPairsMat(diag(true(k,1)))/2;
        nExistPairsMat(diag(true(k,1))) = nExistPairsMat(diag(true(k,1)))/2;
    end

    % Calculate scaled densities of blocks
    newDensMat = nNewEdgesMat./nNewPairsMat;
    newDensMat(nNewPairsMat==0) = emptyVal;
    existDensMat = nExistEdgesMat./nExistPairsMat;
    existDensMat(nExistPairsMat==0) = emptyVal;
    % Set block densities for empty blocks or blocks with densities too
    % close to 0 or 1
    newDensMat(newDensMat < obsClip) = obsClip;
    newDensMat(newDensMat > 1-obsClip) = 1-obsClip;
    existDensMat(existDensMat < obsClip) = obsClip;
    existDensMat(existDensMat > 1-obsClip) = 1-obsClip;

    if directed == true
        p = k^2;
    else
        p = k*(k+1)/2;
    end
    
    % Convert all quantities to vector representation
    nNewEdges(1:p,t) = blockmat2vec(nNewEdgesMat,directed);
    nNewPairs(1:p,t) = blockmat2vec(nNewPairsMat,directed);
    nExistEdges(1:p,t) = blockmat2vec(nExistEdgesMat,directed);
    nExistPairs(1:p,t) = blockmat2vec(nExistPairsMat,directed);
    newDens(1:p,t) = blockmat2vec(newDensMat,directed);
    existDens(1:p,t) = blockmat2vec(existDensMat,directed);
end

end

