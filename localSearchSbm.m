function [class,theta,logLik,Opt] = localSearchSbm(adj,k,classInit,Opt)
%localSearchSbm Fit a posteriori SBM using local search
%   [class,theta,logLik,Opt] = localSearchSbm(adj,k,classInit,Opt)
%
%   localSearchSbm computes parameter estimates for the a posteriori
%   stochastic block model (SBM) using a local search (hill climbing)
%   algorithm over the class memberships to reach a local maximum.
%
%   Inputs:
%   adj - Graph adjacency matrix (binary with no self-edges; can be directed,
%         i.e. adj(i,j) = 1 denotes edge from i to j, and adj(i,j) = 0 denotes
%         absence of edge from i to j)
%   k - Number of classes
%   classInit - Initial class membership vector to begin local search
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%   'maxIter' - Maximum number of local search iterations to perform [ 100 ]
%   'output' - Level of output to print to command window. Higher values
%              result in more output, and 0 results in no output. Set to 2
%              or higher to see the log-likelihood at each iteration of the
%              local search. [ 0 ]
%
%   Outputs:
%   class - Estimated class membership vector
%   theta - Estimated matrix of edge probabilities between blocks
%   logLik - Log-likelihood of parameter estimates
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed','maxIter','output'};
defaultValues = {false,100,0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
maxIter = Opt.maxIter;
output = Opt.output;

if directed == true
    p = k^2;
else
    p = k*(k+1)/2;
end

n = size(adj,1);
class = classInit;  % Estimated class labels

% Compute probability of initial solution as starting point
[theta,logLik,Opt] = estimateSbmProb(adj,class,Opt);

% Perform local search (hill climbing) over class memberships to find a
% local maximum
for iter = 1:maxIter
    if output > 1
        disp(['Iteration ' int2str(iter) ': Log-likelihood ' num2str(logLik)])
    end
    
    % Compute probability for each neighboring solution (each solution that
    % involves changing the class of a single node)
    logLikNb = -Inf*ones(n,k);
    thetaNb = zeros(p,n,k);
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
            [thetaNb(:,iNode,iClass),logLikNb(iNode,iClass)] ...
                = estimateSbmProb(adj,classCurr,Opt);
        end
    end
    
    % Find best neighboring solution
    [logLikNb,idxNb] = max(logLikNb(:));
    [rowNb,colNb] = ind2sub([n k],idxNb);
    thetaNb = thetaNb(:,rowNb,colNb);
    classNb = classNb(:,rowNb,colNb);
    
    % If current best solution among all neighbors is better than the best
    % solution obtained so far, then continue; otherwise, we have reached a
    % local maximum so terminate
    if logLikNb <= logLik
        if output > 1
            disp(['Iteration ' int2str(iter+1) ' (terminated): ' ...
                'Best log-likelihood ' num2str(logLikNb)])
        end
        break
    else
        logLik = logLikNb;
        theta = thetaNb;
        class = classNb;
    end
end

if iter == maxIter
    warning('Maximum number of local search iterations reached')
end

end
