function [blockDens,nEdges,nPairs,Opt] = calcBlockDens(adj,class,Opt)
%calcBlockDens Calculate densities of adjacency matrix blocks
%   [blockDens,nEdges,nPairs,Opt] = calcBlockDens(adj,class,Opt)
%   
%   calcBlockDens calculates the densities of blocks in adjacency matrix
%   adj defined by the classes specified in class membership vector class.
%   adj can also be a sequence of adjacency matrices indexed by the third
%   dimension, where the class memberships given by the i-th column of
%   class are used for the i-th slice of adj.
%
%   Inputs:
%   adj - n x n adjacency matrix of graph, where n denotes the number of
%         nodes. Can also be an n x n x tMax array where each slice of
%         adj is an adjacency matrix.
%   class - Length n vector containing class membership of each node
%           (scalar from 1 to the number of classes k). Can also be an
%           n x tMax matrix where each column is a class membership vector.
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%   'emptyVal' - Density value to set for empty blocks, which occur when a
%                class has no nodes [ 1e-3 ]
%   'obsClip' - How much to clip the range of the estimates of edge
%               probabilities away from the boundaries of 0 and 1, for
%               which the logit is undefined. All entries smaller than
%               obsClip and larger than 1-obsClip are clipped. [ 1e-3 ]
%   'nClasses' - Number of classes [ max(class(:)) ]
%
%   Outputs:
%   blockDens - Length p vector of block densities where p = k*(k+1)/2 for
%               undirected graphs and p = k^2 for directed graphs. If adj
%               is a 3-D array, blockDens will be a matrix where each
%               column corresponds to block densities from each slice of
%               adj.
%   nEdges - Length p vector containing the number of observed edges in
%            each block (or a matrix if adj is a 3-D array)
%   nPairs - Length p vector containing the number of pairs (possible
%            edges) in each block (or a matrix if adj is a 3-D array)
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

nEdges = zeros(pMax,tMax); % Number of edges in each block
% Number of pairs of nodes (number of possible edges) in each block
nPairs = zeros(pMax,tMax);
% Density (number of edges divided by possible edges) of each block
blockDens = zeros(pMax,tMax);

% Identify active nodes at each time step
nodeActive = isNodeActive(adj);

% Compute block densities at each time step
for t = 1:tMax
    adjCurr = adj(nodeActive(:,t),nodeActive(:,t),t);
    cCurr = class(nodeActive(:,t),t);
    k = max(cCurr);
	classSizes = histc(cCurr,1:k);
    
    % Compute numbers of edges in each block
    nEdgesCurrMat = zeros(k,k);
    for c1 = 1:k
		inC1 = (cCurr == c1);
        for c2 = 1:k
			inC2 = (cCurr == c2);
			nEdgesCurrMat(c1,c2) = sum(sum(adjCurr(inC1,inC2)));
        end
    end
    
	nPairsCurrMat = classSizes*classSizes';
    % Reduce the number of possible pairs for diagonal blocks since no
    % self-edges are permitted
	nPairsCurrMat(diag(true(k,1))) = nPairsCurrMat(diag(true(k,1))) ...
        - classSizes;
    
    % For undirected graphs, halve the number of edges and pairs along
    % diagonal blocks
    if directed == false
        nEdgesCurrMat(diag(true(k,1))) = nEdgesCurrMat(diag(true(k,1)))/2;
        nPairsCurrMat(diag(true(k,1))) = nPairsCurrMat(diag(true(k,1)))/2;
    end
    
    blockDensCurrMat = nEdgesCurrMat./nPairsCurrMat;
    % Set block densities for empty blocks or blocks with densities too
    % close to 0 or 1
    blockDensCurrMat(nPairsCurrMat==0) = emptyVal;
    blockDensCurrMat(blockDensCurrMat < obsClip) = obsClip;
    blockDensCurrMat(blockDensCurrMat > 1-obsClip) = 1-obsClip;
    
    if directed == true
        p = k^2;
    else
        p = k*(k+1)/2;
    end
    
    nEdges(1:p,t) = blockmat2vec(nEdgesCurrMat,directed);
    nPairs(1:p,t) = blockmat2vec(nPairsCurrMat,directed);
    blockDens(1:p,t) = blockmat2vec(blockDensCurrMat,directed);
end

Opt.nPairs = nPairs;
