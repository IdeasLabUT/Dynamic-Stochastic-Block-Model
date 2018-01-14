function [class,psi,psiCov,cpuTimes,Opt] = ekfDsbmLocalSearch(adj,k, ...
    stateTrans,transCov,obsCov,initMean,initCov,Opt)
%ekfDsbmLocalSearch EKF with local search for a posteriori dynamic SBM
%   [class,psi,psiCov,cpuTimes,Opt] = ekfDsbmLocalSearch(adj,k,stateTrans, ...
%       transCov,obsCov,initMean,initCov,Opt)
%
%   Inputs:
%   adj - 3-D array of graph adjacency matrices, where each slice along the
%         third dimension denotes the adjacency matrix at time t. Each
%         adjacency matrix is binary with no self-edges and can be directed,
%         i.e. w(i,j,t) = 1 denotes an edge from i to j at time t, and
%         w(i,j,t) = 0 denotes the absence of an edge from i to j at time t.
%   k - Number of classes
%   stateTrans - State transition matrix applied to previous state at time t-1
%   transCov - Process noise covariance matrix driving state dynamics
%   obsCov - Covariance matrix of observation noise. Set to [] if unknown,
%            and a plug-in estimate based on the EKF prediction of the
%            current states E[ psi(t) | y(1:t-1) ] will be used.
%   initMean - Mean of initial state. Set to [] if no prior knowledge; in
%              this case a diffuse prior will be used, i.e. the initial
%              mean will be taken to be logit( y(1) ). initCov should be
%              set as described below for unknown initial covariance.
%   initCov - Covariance matrix of initial state. If unknown, set to a
%             diagonal matrix with large diagonal elements, e.g. 100.
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%   'model' - Model index vector for matrices stateTrans, transCov, and
%             obsCov Sigma. If these matrices are time-varying, then
%             model(t) = m specifies that the matrix to use at time t
%             should be the m-th slice, i.e. stateTrans(:,:,m),
%             transCov(:,:,m), and obsCov(:,:,m). [ all ones vector ]
%   'classInit' - Class estimates to initialize the local search at t=1.
%                 If not specified (set to []), spectral clustering will
%                 be used for the initialization [ [] ]
%   'output' - Level of output to print to command window. Higher values
%              result in more output, and 0 results in no output. Set to 1
%              or higher to display the computation time at each time step.
%              [ 0 ]
%
%   Outputs:
%   class - Matrix of class membership estimates where each column denotes the
%           estimate at at time t
%   psi - Matrix of state estimates where each column denotes the estimate
%         at time t
%   psiCov - Covariance matrices of state estimates where each slice denotes
%            the estimate at time t
%   cpuTimes - Vector of CPU times for each time step t
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

[n,~,tMax] = size(adj);
p = size(stateTrans,1);

% Set defaults for optional parameters if necessary
defaultFields = {'directed','model','classInit','output'};
defaultValues = {false,ones(1,tMax),[],0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
model = Opt.model;
classInit = Opt.classInit;
output = Opt.output;

class = zeros(n,tMax);
psi = zeros(p,tMax);
psiCov = zeros(p,p,tMax);

logistic = @(x) 1./(1+exp(-x));
logit = @(x) log(x)-log(1-x);

% Identify active nodes (nodes that are connected to at least one other
% node) at each time step
nodeActive = isNodeActive(adj);

% Adjacency matrix of active nodes at initial time
adjCurr = adj(nodeActive(:,1),nodeActive(:,1),1);
% If initialization for class estimates at t=1 are not specified, use
% spectral clustering estimate as initialization
if isempty(classInit)
    [classInit,~,~,Opt] = spectralClusterSbm(adjCurr,k,Opt);
end

% Diffuse prior: use initialization for class estimates to estimate block
% densities at t=1. Use these block densities as mean of predicted state so
% that EKF linearization is close to the right point.
if isempty(initMean)
    [initDens,~,~,Opt] = calcBlockDens(adjCurr,classInit,Opt);
    initMean = logit(initDens);
end

% Check if we need to estimate observation noise covariance
if isempty(obsCov)
    obsCov = zeros(0,0,tMax);
end

cpuTimes = zeros(1,tMax);
for t = 1:tMax
    tic
    m = model(t);
    
    % Predict step
    if t == 1
        % Use initial mean E[ psi(0) ] and covariance Cov( psi(0) ) to
        % generate predicted state at time t=1
        [psiPred,psiPredCov,Opt] = ekfPredict(initMean,initCov, ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    else
        [psiPred,psiPredCov] = ekfPredict(psi(:,t-1),psiCov(:,:,t-1), ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    end
    
    % Update step
    if t == 1
        [class(nodeActive(:,t),t),psi(:,t),psiCov(:,:,t)] ...
            = localSearchDsbm(adjCurr,k,classInit,psiPred,psiPredCov, ...
            obsCov(:,:,t),Opt);
    else
        % Initialize class memberships at current time for all nodes to
        % memberships at last active time for each node
        class(:,t) = class(:,t-1);
        
        % Adjacency matrix and class vector of active nodes
        adjCurr = adj(nodeActive(:,t),nodeActive(:,t),t);
        classInit = class(nodeActive(:,t),t);
        
        % Assign new nodes to most similar class in terms of block
        % probabilities
        newNode = classInit==0;
        nNewNodes = sum(newNode);
        fracFromNewNodes = zeros(nNewNodes,k);
        fracToNewNodes = zeros(k,nNewNodes);
        for iClass = 1:k
            nodeInClass = classInit==iClass;
            nNodesInClass = sum(nodeInClass);
            if nNodesInClass == 0
                continue
            end
            fracFromNewNodes(:,iClass) = sum(adjCurr(newNode,nodeInClass),2) ...
                / nNodesInClass;
            fracToNewNodes(iClass,:) = sum(adjCurr(nodeInClass,newNode),1) ...
                / nNodesInClass;
        end
        distClass = pdist2(fracFromNewNodes,logistic(blockvec2mat(psiPred, ...
            directed))) + pdist2(fracToNewNodes',logistic(blockvec2mat ...
            (psiPred,directed))');
        [~,bestClass] = min(distClass,[],2);
        classInit(newNode) = bestClass;
        
        % Update step
        [class(nodeActive(:,t),t),psi(:,t),psiCov(:,:,t)] ...
            = localSearchDsbm(adjCurr,k,classInit,psiPred,psiPredCov, ...
            obsCov(:,:,t),Opt);
    end
    cpuTimes(t) = toc;
    
    if output > 0
        disp(['t = ' int2str(t) ': ' num2str(cpuTimes(t)) ' seconds'])
    end
end

end

