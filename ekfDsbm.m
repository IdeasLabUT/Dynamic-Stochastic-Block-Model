function [psi,psiCov,logLik,Opt] = ekfDsbm(y,stateTrans,transCov,obsCov, ...
    initMean,initCov,Opt)
%ekfDsbm Extended Kalman filter for dynamic stochastic block model
%   [psi,psiCov,logLik,Opt] = ekf(y,stateTrans,transCov,obsCov,initMean, ...
%       initCov,Opt)
%
%   Inputs:
%   y - Matrix of observations where each column denotes the observation at
%       a particular time t
%   stateTrans - State transition matrix applied to previous state at time t-1
%   transCov - Process noise covariance matrix driving state dynamics
%   obsCov - Covariance matrix of observation noise. Set to [] if unknown,
%           and a plug-in estimate based on the EKF prediction of the
%           current states E[ psi(t) | y(1:t-1) ] will be used.
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
%   'model' - Model index vector for matrices stateTrans and transCov. If
%             these matrices are time-varying, then model(t) = m specifies
%             that the matrix to use at time t should be the m-th slice,
%             i.e. stateTrans(:,:,m) and transCov(:,:,m). [ all ones
%             vector ]
%   'nPairs' - Matrix of number of pairs (possible edges) in each block
%              where each column denotes a time step t. This is necessary
%              only if estimating obsCov, in which case obsCov should be
%              set to []. [ [] ]
%
%   Outputs:
%   psi - Matrix of state estimates where each column denotes the estimate
%         at time t, E[ psi(t) | y(1:t) ]
%   psiCov - Covariance matrices of state estimates where each slice denotes
%            the estimate at time t, Cov( psi(t) | y(1:t) )
%   logLik - Log-likelihood of innovation sequence where each entry denotes
%            log f( y(t) | y(1:t-1) )
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

[p,tMax] = size(y);

% Set defaults for optional parameters if necessary
defaultFields = {'model','nPairs'};
defaultValues = {ones(1,tMax),[]};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
model = Opt.model;
nPairs = Opt.nPairs;

psi = zeros(p,tMax);
psiCov = zeros(p,p,tMax);
logLik = zeros(1,tMax);

logistic = @(x) 1./(1+exp(-x));
logit = @(x) log(x)-log(1-x);

% Check if we need to estimate observation noise covariance
if isempty(obsCov)
    % Need also number of possible edges in order to estimate observation
    % noise variances
    if isempty(nPairs)
        error('Either obsCov or nPairs needs to be specified')
    end
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

% Diffuse prior: use first observation as mean of predicted state so that
% EKF linearization is close to the right point
if isempty(initMean)
    initMean = logit(y(:,1));
end

for t = 1:tMax
    m = model(t);
    
    % Predict step
    if t == 1
        % Use initial mean E[ psi(0) ] and covariance Cov( psi(0) ) to
        % generate predicted state at time t=1
        [psiPred,psiPredCov,Opt] = ekfPredict(initMean,initCov, ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    else
        [psiPred,psiPredCov,Opt] = ekfPredict(psi(:,t-1),psiCov(:,:,t-1), ...
            stateTrans(:,:,m),transCov(:,:,m),Opt);
    end
    
    % If necessary, estimate observation noise covariance matrix Sigma
    % using psiPred
    if estimateSigma == true
        nPairsCurr = nPairs(:,t);
        % If a group is empty (no observations), assume just a single
        % observation
        nPairsCurr(nPairsCurr==0) = 1;
        obsVarCurr = logistic(psiPred).*(1-logistic(psiPred)) ./ nPairsCurr;
        obsCov(:,:,t) = diag(obsVarCurr);
    end
    
    % Update step
    [psi(:,t),psiCov(:,:,t),logLik(t),Opt] = ekfUpdate(y(:,t),psiPred, ...
        psiPredCov,obsCov(:,:,t),Opt);
end

end

