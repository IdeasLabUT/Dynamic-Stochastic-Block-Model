function [class,psi,psiCov,logPost,Opt] = localSearchDsbm(adj,k,classInit, ...
    psiPred,psiCovPred,obsCov,Opt)
%localSearchDsbm Fit a posteriori dynamic SBM using local search
%   [class,psi,psiCov,logPost,Opt] = localSearchDsbm(adj,k,classInit, ...
%       psiPred,psiCovPred,obsCov,Opt)
%
%   Computes state estimates for the a posteriori dynamic stochastic block
%   model (SBM) using a local search (hill climbing) algorithm over the
%   class memberships combined with the extended Kalman filter to maximize
%   the posterior probability of the states.
%
%   Inputs:
%   adj - Graph adjacency matrix (binary with no self-edges; can be directed,
%         i.e. w(i,j) = 1 denotes edge from i to j, and w(i,j) = 0 denotes
%         absence of edge from i to j)
%   k - Number of classes
%   classInit - Initial class membership vector to begin local search
%   psiPred - Predicted state estimate given observations up to time t-1
%             E[ psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state estimate
%                Cov( psi(t) | y(1:t-1) )
%   obsCov - Covariance matrix of observation noise (set to [] if unknown)
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'maxIter' - Maximum number of local search iterations to perform [ 100 ]
%   'output' - Level of output to print to command window. Higher values
%              result in more output, and 0 results in no output. Set to 2
%              or higher to see the log of the posterior probability at
%              each iteration of the local search. [ 0 ]
%
%   Outputs:
%   class - Estimated class membership vector
%   psi - Updated state estimates E[ psi(t) | y(1:t) ]
%   psiCov - Updated state covariance matrix Cov( psi(t) | y(1:t) )
%   logPost - Log of posterior probability of the state estimates
%             log f( psi(t) | adj(t) )
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'maxIter','output'};
defaultValues = {100,0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
maxIter = Opt.maxIter;
output = Opt.output;

n = size(adj,1);
p = length(psiPred);
class = classInit;  % Estimated class labels

% Compute probability of initial solution as starting point
[psi,psiCov,logPost,Opt] = estimateDsbmProb(adj,class,psiPred, ...
    psiCovPred,obsCov,Opt);

% Perform local search (hill climbing) over class memberships to find a
% local maximum
for iter = 1:maxIter
    if output > 1
        disp(['Iteration ' int2str(iter) ': Log-posterior ' num2str(logPost)])
    end
    
    % Compute probability for each neighboring solution (each solution that
    % involves changing the class of a single node)
    logPostNb = -Inf*ones(n,k);
    psiNb = zeros(p,n,k);
    psiCovNb = zeros(p,p,n,k);
    classNb = zeros(n,n,k);
    parfor iNode = 1:n
        for iClass = 1:k
            % Don't re-evaluate current best solution
            if class(iNode) == iClass
                continue
            end
            
            % Class assignment currently being evaluated
            classCurr = class; %#ok<PFBNS>
            % Move node to class iClass and compute log-likelihood
            classCurr(iNode) = iClass;
            classNb(:,iNode,iClass) = classCurr;
            [psiNb(:,iNode,iClass),psiCovNb(:,:,iNode,iClass), ...
                logPostNb(iNode,iClass)] = estimateDsbmProb(adj, ...
                classCurr,psiPred,psiCovPred,obsCov,Opt);
        end
    end
    
    % Find best neighboring solution
    [logPostNb,idxNb] = max(logPostNb(:));
    [rowNb,colNb] = ind2sub([n k],idxNb);
    psiNb = psiNb(:,rowNb,colNb);
    psiCovNb = psiCovNb(:,:,rowNb,colNb);
    classNb = classNb(:,rowNb,colNb);
    
    % If current best solution among all neighbors is better than the best
    % solution obtained so far, then continue; otherwise, we have reached a
    % local maximum so terminate
    if logPostNb <= logPost
        if output > 1
            disp(['Iteration ' int2str(iter+1) ' (terminated): ' ...
                'Best log-posterior ' num2str(logPostNb)])
        end
        break
    else
        logPost = logPostNb;
        psi = psiNb;
        psiCov = psiCovNb;
        class = classNb;
    end
end

if iter == maxIter
    warning('Maximum number of local search iterations reached')
end

end
