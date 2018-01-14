function [psi,psiCov,logPost,Opt] = estimateDsbmProb(adj,class,psiPred, ...
    psiPredCov,obsCov,Opt)
%estimateDsbmProb Estimate edge probabilities for dynamic SBM
%   [psi,psiCov,logPost,Opt] = estimateDsbmProb(adj,class,psiPred, ...
%       psiPredCov,obsCov,Opt)
%
%   Inputs:
%   adj - Graph adjacency matrix (binary with no self-edges; can be directed,
%         i.e. adj(i,j) = 1 denotes edge from i to j, and adj(i,j) = 0 denotes
%         absence of edge from i to j)
%   class - Class membership vector
%   psiPred - Predicted state estimate given observations up to time t-1
%             E[ psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state estimate
%                Cov( psi(t) | y(1:t-1) ) 
%   obsCov - Covariance matrix of observation noise (set to [] if unknown)
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Outputs:
%   psi - Updated state estimate E[ psi(t) | y(1:t) ]
%   psiCov - Updated state covariance matrix Cov( psi(t) | y(1:t) )
%   logPost - Log of posterior probability log f( psi(t) | adj(t) )
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

logistic = @(x) 1./(1+exp(-x));

% Compute block statistics
[y,nEdges,nPairs,Opt] = calcBlockDens(adj,class,Opt);

% If necessary, estimate observation noise covariance matrix Sigma
% using psiPred
if isempty(obsCov)
    nPairs(nPairs==0) = 1;
    obsCov = diag(logistic(psiPred).*(1-logistic(psiPred))./nPairs);
end

% Run EKF to obtain estimate of psi(t)
[psi,psiCov,~,Opt] = ekfUpdate(y,psiPred,psiPredCov,obsCov,Opt);

% Calculate log of prior probability f( psi(t) | y(1:t-1) )
logPrior = log(mvnpdf(psi,psiPred,psiPredCov));

% Calculate log-likelihood f( adj(t) | psi(t) )
theta = logistic(psi);
logLik = sum(nEdges.*log(theta) + (nPairs-nEdges).*log(1-theta));

% Calculate log of posterior probability f( psi(t) | adj(t) )
logPost = logPrior + logLik;

end

