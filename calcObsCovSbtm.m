function [obsCov,varNewMat,varExistMat,Opt] = calcObsCovSbtm(adj, ...
    class,scaleFactors,probNew,probExist,nNewPairs,nExistPairs,Opt)
%calcObsCovSbtm Calculate observation covariance matrix for SBTM
%   [obsCov,varNewMat,varExistMat,Opt] = calcObsCovSbtm(adj, ...
%       class,scaleFactors,probNew,probExist,nNewPairs,nExistPairs,Opt)
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
%   probNew - p x tMax matrix of probabilities of forming new edges in
%             blocks where p = k*(k+1)/2 for undirected graphs and p = k^2
%             for directed graphs. Each column corresponds to a vector
%             representation of new edge probabilities for blocks from each
%             slice of adj.
%   probExist - p x tMax matrix of probabilities of existing edges
%               re-occurring in blocks.
%   nNewPairs - p x tMax matrix containing the number of new pairs
%               (possible new edges) in blocks
%   nExistPairs - p x tMax matrix containing the number of re-occuring
%                 pairs (possible re-occurring edges) in blocks
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%   'nClasses' - Number of classes [ max(class(:)) ]
%
%   Outputs:
%   obsCov - 2p x 2p x tMax array where each slice denotes the covariance
%            matrix of the observation (scaled densities of new and
%            re-occurring edges in blocks stacked into a single vector).
%   varNewMat - k x k x tMax array where each slice is a matrix denoting
%               the variance of the scaled density of new edges in a
%               particular block
%   varExistMat - k x k x tMax array where each slice is a matrix denoting
%                 the variance of the scaled density of re-occurring edges
%                 in a particular block
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed','nClasses'};
defaultValues = {false,[]};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
nClasses = Opt.nClasses;

[n,~,tMax] = size(adj);  % Number of nodes and snapshots (time steps)
% Maximum number of classes over all time steps
if isempty(nClasses)
    k = max(class(:));
else
    k = nClasses;
end
% Length of density vector depends on whether graphs are directed or
% undirected
if directed == true
    p = 2*k^2;
else
    p = 2*k*(k+1)/2;
end
% If class membership is specified by a vector, repeat it into a matrix
% with a column vector for each time
if isvector(class)
    class = repmat(reshape(class,n,1),1,tMax);
end

obsCov = zeros(p,p,tMax);
varNewMat = zeros(k,k,tMax);
varExistMat = zeros(k,k,tMax);

for t = 2:tMax
    varNewCurrMat = zeros(k,k);
    varExistCurrMat = zeros(k,k);
    nNewPairsCurrMat = blockvec2mat(nNewPairs(:,t),directed);
    nExistPairsCurrMat = blockvec2mat(nExistPairs(:,t),directed);
    scaleFactorsCurr = scaleFactors(:,:,t);
    % Binary masks denoting edges and non-edges at previous time
    zeroMask = (adj(:,:,t-1)==0);
    oneMask = ~zeroMask;
    % Remove diagonal from set of non-edges since we are not allowing
    % self-edges
    zeroMask(diag(true(n,1))) = false;
    % Probabilities of forming new edges in blocks
    probNewCurrMat = blockvec2mat(probNew(:,t),directed);
    % Probabilities of edges re-occurring in blocks
    probExistCurrMat = blockvec2mat(probExist(:,t),directed);
    
    % Compute observation noise variances
    for c1 = 1:k
        for c2 = 1:k
            blockMask = false(n,n);
            blockMask(class(:,t)==c1,class(:,t)==c2) = true;
            zeroBlock = zeroMask & blockMask;
            oneBlock = oneMask & blockMask;

            % For empty blocks, arbitrarily set variance of observation to 1
            if nNewPairsCurrMat(c1,c2) == 0
                varNewCurrMat(c1,c2) = 1;
            else
                sumInvScaleFactors = sum(1./scaleFactorsCurr(zeroBlock));
                if (c1 == c2) && (directed == false)
                    % Reduce number of pairs by factor of 2 if undirected
                    % graph and diagonal block
                    sumInvScaleFactors = sumInvScaleFactors/2;
                end
                varNewCurrMat(c1,c2) = (probNewCurrMat(c1,c2) ...
                    * sumInvScaleFactors - nNewPairsCurrMat(c1,c2) ...
                    * probNewCurrMat(c1,c2)^2) / nNewPairsCurrMat(c1,c2)^2;
            end
            if nExistPairsCurrMat(c1,c2) == 0
                varExistCurrMat(c1,c2) = 1;
            else
                sumInvScaleFactors = sum(1./scaleFactorsCurr(oneBlock));
                if (c1 == c2) && (directed == false)
                    % Reduce number of pairs by factor of 2 if undirected
                    % graph and diagonal block
                    sumInvScaleFactors = sumInvScaleFactors/2;
                end
                varExistCurrMat(c1,c2) = (probExistCurrMat(c1,c2) ...
                    * sumInvScaleFactors - nExistPairsCurrMat(c1,c2) ...
                    * probExistCurrMat(c1,c2)^2) / nExistPairsCurrMat(c1,c2)^2;
            end
        end
    end
    
    varNewMat(:,:,t) = varNewCurrMat;
    varExistMat(:,:,t) = varExistCurrMat;
    obsCov(:,:,t) = diag([blockmat2vec(varNewCurrMat,directed); ...
        blockmat2vec(varExistCurrMat,directed)]);
end
    
end

