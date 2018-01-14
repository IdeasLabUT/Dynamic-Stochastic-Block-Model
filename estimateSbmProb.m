function [theta,logLik,Opt] = estimateSbmProb(adj,class,Opt)
%estimateSbmProb Compute maximum-likelihood estimate of SBM probabilities
%   [theta,logLik,Opt] = estimateSbmProb(adj,class,Opt)
%
%   estimateSbmProb computes the maximum-likelihood (ML) estimate of the
%   edge probabilities between blocks in a stochastic block model (SBM)
%   given a graph adjacency matrix adj and class membership vector class.
%   The class memberships are taken as fixed; thus the ML estimate is only
%   over the edge probabilities.
%
%   Inputs:
%   adj - n x n adjacency matrix of graph, where n denotes the number of
%         nodes.
%   class - Length n vector containing class membership of each node
%           (scalar from 1 to the number of blocks k).
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Outputs:
%   theta - Length p vector of ML estimate of SBM edge probabilities,
%           where p = k*(k+1)/2 for undirected graphs and p = k^2 for
%           directed graphs. theta can be converted into a block matrix
%           using blockvec2mat.
%   logLik - Log-likelihood of the ML estimate.
%   Opt - Updated struct of optional parameter values

% Author: Kevin Xu

% Calculate edge probabilities between blocks and convert to vector form
% (applicable to both undirected and directed graphs)
[theta,nEdges,nPairs,Opt] = calcBlockDens(adj,class,Opt);

% Compute log-likelihood and return matrix of edge probabilities between
% blocks
logLik = sum(nEdges.*log(theta) + (nPairs-nEdges).*log(1-theta));

end