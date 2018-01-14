function [psi,psiCov,logLik,scaleFactors,newExistDens,probMarg,Opt] ...
    = ekfSbtm(adj,class,stateTrans,transCov,obsCov,initMean,initCov,Opt)
%ekfSbtm Extended Kalman filter for stochastic block transition model
%   [psi,psiCov,logLik,scaleFactors,newExistDens,probMarg,Opt] ...
%       = ekfSbtm(adj,class,stateTrans,transCov,obsCov,initMean,initCov,Opt)
%
%   Inputs:
%   adj - 3-D array of graph adjacency matrices, where each slice along the
%         third dimension denotes the adjacency matrix at time t. Each
%         adjacency matrix is binary with no self-edges and can be directed,
%         i.e. w(i,j,t) = 1 denotes an edge from i to j at time t, and
%         w(i,j,t) = 0 denotes the absence of an edge from i to j at time t.
%   class - Matrix of nodes' class memberships where each column denotes the
%           class memberships at at time t
%   stateTrans - State transition matrix applied to previous state at time t-1
%   transCov - Process noise covariance matrix driving state dynamics
%   obsCov - Covariance matrix of observation noise. Set to [] if unknown,
%           and a plug-in estimate based on the EKF prediction of the
%           current states E[ psi(t) | adj(1:t-1) ] will be used.
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
%   'model' - Model index vector for matrices stateTrans and transCov. If
%             these matrices are time-varying, then model(t) = m specifies
%             that the matrix to use at time t should be the m-th slice,
%             i.e. stateTrans(:,:,m) and transCov(:,:,m). [ all ones
%             vector ]
%   'nScaleIter' - Number of iterations to use when estimating matrix of
%                  scaling factors. At the first iteration, the EKF
%                  predicted state is used; at subsequent iterations, the
%                  EKF updated state is used. [ 1 ]
%   'output' - Level of output to print to command window. Higher values
%              result in more output, and 0 results in no output. Set to 1
%              or higher to display the computation time at each time step.
%              [ 0 ]
%
%   Outputs:
%   psi - Matrix of state estimates where each column denotes the estimate
%         at time t, E[ psi(t) | adj(1:t) ]
%   psiCov - Covariance matrices of state estimates where each slice denotes
%            the estimate at time t, Cov( psi(t) | adj(1:t) )
%   logLik - Log-likelihood of innovation sequence where each entry denotes
%            log f( adj(t) | adj(1:t-1) )
%   scaleFactors - 3-D array where each slice denotes the estimated matrix
%                  of scaling factors at time t
%   newExistDens - Matrix of sample means of scaled adjacency matrix blocks,
%                  where each column denotes observations at time t
%   probMarg - Matrix of estimated marginal edge probabilities for blocks,
%              where each column denotes estimates at time t in vector form
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

[n,~,tMax] = size(adj);
k = max(class(:));

% Set defaults for optional parameters if necessary
defaultFields = {'directed','model','nScaleIter','output'};
defaultValues = {false,ones(1,tMax),1,0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
model = Opt.model;
nScaleIter = Opt.nScaleIter;
output = Opt.output;

if directed == true
    p = 2*k^2;
else
    p = 2*k*(k+1)/2;
end

psi = zeros(p,tMax);
psiCov = zeros(p,p,tMax);
logLik = zeros(1,tMax);
scaleFactors = zeros(n,n,tMax);
probMarg = zeros(p/2,tMax);

logistic = @(x) 1./(1+exp(-x));
logit = @(x) log(x)-log(1-x);

% Check if we need to estimate observation noise covariance
if isempty(obsCov)
    % Need also number of possible edges in order to estimate observation
    % noise variances
%     if isempty(nPairs)
%         error('Either obsCov or nPairs needs to be specified')
%     end
    estimateSigma = true;
    obsCov = zeros(p,p,tMax);
else
    estimateSigma = false;
    % If observation noise covariance is given as a single matrix, repeat
    % it over all time steps
    if size(obsCov,3) == 1
        obsCov = repmat(obsCov,[1 1 tMax]);
    end
end

% Diffuse prior: apply class estimate at t=1 to t=2 also and use densities
% of new and existing edges as mean of predicted state
if isempty(initMean)
    [newDens,existDens,~,~,~,~,Opt] = calcNewExistDens(adj(:,:,[1 2]), ...
        class(:,[1 1]),cat(3,ones(n),ones(n)),Opt);
    initMean = logit([newDens(:,2); existDens(:,2)]);
end

% Initial time step: compute block densities (ML estimates of marginal
% block edge probabilities)
[probMarg(:,1),~,~,Opt] = calcBlockDens(adj(:,:,1),class(:,:,1),Opt);
probMargCurrMat = blockvec2mat(probMarg(:,1),directed);

% Subsequent time steps: estimate scaling factors then run EKF on the
% sample means of scaled adjacencies in blocks
newExistDens = zeros(p,tMax);
for t = 2:tMax
    m = model(t);
    
    % Predict step
    if t == 2
        % Use initial mean E[ psi(0) ] and covariance Cov( psi(0) ) to
        % generate predicted state at time t=2 (first time step where
        % transitions are possible)
        [psiPred,psiCovPred,Opt] = ekfPredict(initMean,initCov, ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    else
        [psiPred,psiCovPred,Opt] = ekfPredict(psi(:,t-1),psiCov(:,:,t-1), ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    end
    
    % Iteratively estimate scale factors and states, beginning with EKF
    % predicted state
    psi(:,t) = psiPred;
    for iScale = 1:nScaleIter
        % Compute scale factors and block statistics
        thetaNewMat = blockvec2mat(logistic(psi(1:p/2,t)),directed);
        thetaExistMat = blockvec2mat(logistic(psi(p/2+1:end,t)),directed);
        scaleFactors(:,:,t) = calcSbtmScaleFactors(adj(:,:,t-1),class(:,t), ...
            class(:,t-1),thetaNewMat,thetaExistMat,probMargCurrMat,directed);
        [newDens,existDens,~,~,nNewPairs,nExistPairs,Opt] ...
            = calcNewExistDens(adj(:,:,[t-1 t]),class(:,[t-1 t]), ...
            scaleFactors(:,:,[t-1 t]),Opt);
        newExistDens(:,t) = [newDens(:,2); existDens(:,2)];

        % If necessary, estimate observation noise covariance matrix Sigma
        % using psiPred
        if estimateSigma == true
            nPairsCurr = [nNewPairs(:,2); nExistPairs(:,2)];
            % If a group is empty (no observations), assume just a single
            % observation
            nPairsCurr(nPairsCurr==0) = 1;
            obsCov(:,:,t) = diag(logistic(psiPred).*(1-logistic(psiPred)) ...
                ./ nPairsCurr);
        end

        % Update step
        [psi(:,t),psiCov(:,:,t),logLik(t),Opt] = ekfUpdate(newExistDens ...
            (:,t),psiPred,psiCovPred,obsCov(:,:,t),Opt);
        
        if output > 0
            disp(['t = ' int2str(t) ' complete'])
        end
    end
    
    % Update marginal probabilities of blocks
    thetaNewMat = blockvec2mat(logistic(psi(1:p/2,t)),directed);
    thetaExistMat = blockvec2mat(logistic(psi(p/2+1:end,t)),directed);
    probMargCurrMat = thetaNewMat.*(1-probMargCurrMat) + thetaExistMat ...
        .*probMargCurrMat;
    probMarg(:,t) = blockmat2vec(probMargCurrMat,directed);
end

end

