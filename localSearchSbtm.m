function [class,psi,psiCov,logPost,cpuTimes,newExistDens,probMarg,Opt] ...
    = localSearchSbtm(adj,adjPrev,k,classPrev,classInit,psiPred, ...
    psiCovPred,probMargPrev,obsCov,Opt)
%localSearchSbtm Fit a posteriori SBTM using local search
%   [class,psi,psiCov,logPost,newExistDens,probMarg,Opt] ...
%       = localSearchSbtm(adj,adjPrev,k,classPrev,classInit,psiPred, ...
%       psiCovPred,probMargPrev,obsCov,Opt)
%
%   Computes state estimates for the a posteriori stochastic block transition
%   model (SBTM) using a local search (hill climbing) algorithm over the
%   class memberships combined with the extended Kalman filter to maximize
%   the posterior probability of the states.
%
%   Inputs:
%   adj - Graph adjacency matrix (binary with no self-edges; can be directed,
%         i.e. w(i,j) = 1 denotes edge from i to j, and w(i,j) = 0 denotes
%         absence of edge from i to j)
%   adjPrev - Graph adjacency matrix at previous time
%   k - Number of classes
%   classPrev - Class membership vector at previous time
%   classInit - Initial class membership vector at current time to begin
%               local search
%   psiPred - Predicted state estimate given observations up to time t-1
%             E[ psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state estimate
%                Cov( psi(t) | y(1:t-1) )
%   probMargPrev - Marginal block edge probability estimates at the previous
%                  time step (used to estimate SBTM scale factors)
%   obsCov - Covariance matrix of observation noise (set to [] if unknown)
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
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
%   logPost - Log of posterior probability of the states
%             log f( psi(t) | adj(t) )
%   newExistDens - Matrix of sample means of scaled adjacency matrix blocks,
%                  where each column denotes observations at time t
%   probMarg - Updated marginal block edge probability estimate at time t,
%              used to estimate SBTM scale factors at future time steps
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed','maxIter','output'};
defaultValues = {false,100,0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
maxIter = Opt.maxIter;
output = Opt.output;

logistic = @(x) 1./(1+exp(-x));

n = size(adj,1);
p = length(psiPred);
class = classInit;  % Estimated class labels

% Compute probability of initial solution as starting point
[psi,psiCov,logPost,newExistDens,scaleFactors,Opt] = estimateSbtmProb(adj, ...
    adjPrev,class,classPrev,psiPred,psiCovPred,probMargPrev,obsCov,[],[],Opt);

cpuTimes = zeros(maxIter,1);

% Perform local search (hill climbing) over class memberships to find a
% local maximum
for iter = 1:maxIter
    if output > 1
        disp(['Iteration ' int2str(iter) ': Log-posterior ' num2str(logPost)])
    end
    
    tIterStart = tic;
    
    % Compute probability for each neighboring solution (each solution that
    % involves changing the class of a single node)
    logPostNb = -Inf*ones(n,k);
    psiNb = zeros(p,n,k);
    psiCovNb = zeros(p,p,n,k);
    classNb = zeros(n,n,k);
    newExistDensNb = zeros(p,n,k);
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
                logPostNb(iNode,iClass),newExistDensNb(:,iNode,iClass)] ...
                = estimateSbtmProb(adj,adjPrev,classCurr,classPrev, ...
                psiPred,psiCovPred,probMargPrev,obsCov,scaleFactors,iNode,Opt);
        end
    end
    
    % Find best neighboring solution
    [logPostNb,idxNb] = max(logPostNb(:));
    [rowNb,colNb] = ind2sub([n k],idxNb);
    psiNb = psiNb(:,rowNb,colNb);
    psiCovNb = psiCovNb(:,:,rowNb,colNb);
    classNb = classNb(:,rowNb,colNb);
    newExistDensNb = newExistDensNb(:,rowNb,colNb);
    
    cpuTimes(iter) = toc(tIterStart);
    
    % If current best solution among all neighbors is better than the best
    % solution obtained so far, then continue; otherwise, we have reached a
    % local maximum so terminate
    if logPostNb <= logPost
        if output > 1
            disp(['Iteration ' int2str(iter+1) ' (terminated): ' ...
                'Best neighboring log-posterior ' num2str(logPostNb)])
        end
        break
    else
        logPost = logPostNb;
        psi = psiNb;
        psiCov = psiCovNb;
        class = classNb;
        newExistDens = newExistDensNb;
        
        % Obtain updated scale factors for best neighboring solution
        thetaNewPredMat = blockvec2mat(logistic(psiPred(1:p/2)),directed);
        thetaExistPredMat = blockvec2mat(logistic(psiPred(p/2+1:end)), ...
            directed);
        probMargPrevMat = blockvec2mat(probMargPrev,directed);
        scaleFactors = updateSbtmScaleFactorsNode(adjPrev,class,classPrev, ...
            thetaNewPredMat,thetaExistPredMat,probMargPrevMat,scaleFactors, ...
            rowNb,directed);
    end    
end

cpuTimes = cpuTimes(1:iter);

% Compute marginal block probabilities for this solution (required
% to calculate SBTM scaling factors at future time steps)
thetaNew = logistic(psi(1:p/2));
thetaExist = logistic(psi(p/2+1:end));
probMarg = thetaNew.*(1-probMargPrev) + thetaExist.*probMargPrev;

if iter == maxIter
    warning('Maximum number of local search iterations reached')
end

end
