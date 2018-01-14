function [class,psi,psiCov,cpuTimes,newExistDens,probMarg,Opt] ...
    = ekfSbtmLocalSearch(adj,k,stateTrans,transCov,obsCov,initMean, ...
    initCov,Opt)
%ekfSbtmLocalSearch EKF with local search for a posteriori SBTM
%   [class,psi,psiCov,cpuTimes,newExistDens,probMarg,Opt] ...
%       = ekfSbtmLocalSearch(adj,k,stateTrans,transCov,obsCov,initMean, ...
%       initCov,Opt)
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
%            current states E[ psi(t) | adj(1:t-1) ] will be used.
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
newExistDens = zeros(p,tMax);
probMarg = zeros(p/2,tMax);
cpuTimes = struct('total',0,'iteration',cell(1,tMax));

logistic = @(x) 1./(1+exp(-x));
logit = @(x) log(x)-log(1-x);

% Identify active nodes at each time step (nodes connected to at least one
% other node)
nodeActive = isNodeActive(adj);

% Adjacency matrix of active nodes at initial time
adjCurr = adj(nodeActive(:,1),nodeActive(:,1),1);

% If initialization for class estimates at t=1 are not specified, use
% spectral clustering estimate as initialization
if isempty(classInit)
    [classInit,~,~,Opt] = spectralClusterSbm(adjCurr,k,Opt);
end

% Initial time step: compute block densities (ML estimates of marginal
% block edge probabilities)
tic
[class(nodeActive(:,1),1),probMarg(:,1),~,Opt] = localSearchSbm(adjCurr, ...
    k,classInit,Opt);
cpuTimes(1).total = toc;
if output > 0
    disp(['t = ' int2str(1) ': ' num2str(cpuTimes(1).total) ' seconds'])
end

% Diffuse prior: apply class estimate at t=1 to t=2 also and use densities
% of new and existing edges as mean of predicted state
if isempty(initMean)
    [newDens,existDens,~,~,~,~,Opt] = calcNewExistDens(adj(:,:,[1 2]), ...
        class(:,[1 1]),cat(3,ones(n),ones(n)),Opt);
    initMean = logit([newDens(:,2); existDens(:,2)]);
end

% Check if we need to estimate observation noise covariance
if isempty(obsCov)
    obsCov = zeros(0,0,tMax);
end

for t = 2:tMax
    tic
    m = model(t);
    
    % Predict step
    if t == 2
        % Use initial mean E[ psi(0) ] and covariance Cov( psi(0) ) to
        % generate predicted state at time t=2
        [psiPred,psiPredCov,Opt] = ekfPredict(initMean,initCov, ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    else
        [psiPred,psiPredCov,Opt] = ekfPredict(psi(:,t-1),psiCov(:,:,t-1), ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    end
    
    % Predicted marginal probability matrix for blocks
    thetaNewPred = logistic(psiPred(1:p/2));
    thetaExistPred = logistic(psiPred(p/2+1:end));
    probMargPred = thetaNewPred.*(1-probMarg(:,t-1)) + thetaExistPred ...
        .* probMarg(:,t-1);
    probMargPredMat = blockvec2mat(probMargPred,directed);
    
    % Initialize class memberships at current time for all nodes to
    % memberships at last active time for each node
    class(:,t) = class(:,t-1);
    
    % Current adjacency matrix entries for currently active nodes
    adjCurr = adj(nodeActive(:,t),nodeActive(:,t),t);
    % Previous adjacency matrix entries for currently active nodes
    adjPrev = adj(nodeActive(:,t),nodeActive(:,t),t-1);
    % Previous class memberships for currently active nodes
    classPrev = class(nodeActive(:,t),t-1);
    % Initialization of class memberships for currently active nodes
    classInit = class(nodeActive(:,t),t);
    
    % Assign new nodes to closest class in terms of marginal probabilities
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
    distClass = pdist2(fracFromNewNodes,probMargPredMat) ...
        + pdist2(fracToNewNodes',probMargPredMat');
    [~,bestClass] = min(distClass,[],2);
    classInit(newNode) = bestClass;
    
    % Update step
    [class(nodeActive(:,t),t),psi(:,t),psiCov(:,:,t),~, ...
        cpuTimes(t).iteration,newExistDens(:,t), ...
        probMarg(:,t),Opt] = localSearchSbtm(adjCurr,adjPrev,k,classPrev, ...
        classInit,psiPred,psiPredCov,probMarg(:,t-1),obsCov(:,:,t),Opt);
    cpuTimes(t).total = toc;
    
    if output > 0
        disp(['t = ' int2str(t) ': ' num2str(cpuTimes(t).total) ' seconds'])
%         disp(['t = ' int2str(t) ': ' num2str(mean(cpuTimesIter{t})) ...
%             ' seconds'])
    end
end

end

