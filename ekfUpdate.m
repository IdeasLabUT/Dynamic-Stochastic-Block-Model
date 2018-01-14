function [psi,psiCov,logLik,Opt] = ekfUpdate(y,psiPred,psiPredCov,obsCov,Opt)
%ekfUpdate Update phase of extended Kalman filter for dynamic SBM
%   [psi,psiCov,logLik,Opt] = ekfUpdate(y,psiPred,psiPredCov,obsCov,Opt)
%
%   Inputs:
%   y - The current observation
%   psiPred - Predicted state estimate given observations up to time t-1,
%             E[psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state estimate
%            Cov( psi(t) | y(1:t-1) ) 
%   obsCov - Covariance matrix of observation noise
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'innovCov' - Innovation covariance [ [] ]
%   'kalmanGain' - Kalman gain matrix [ [] ]
%   'obsClip' - How much to clip the range of the estimates of edge
%               probabilities away from the boundaries of 0 and 1, for
%               which the logit is undefined. All entries smaller than
%               obsClip and larger than 1-obsClip are clipped. [ 1e-3 ]
%   'stateClip' - Value at which to clip state estimates. Estimates that
%                 exceed stateClip in magnitude will be clipped. [ 10 ]
%   'eigShift' - Amount to shift eigenvalues of estimated state covariance
%                if positive definiteness is lost. Set to empty matrix to
%                ignore check for positive definiteness. [ [] ]
%
%   Outputs:
%   psi - Updated state estimate E[ psi(t) | y(1:t) ]
%   psiCov - Updated state covariance matrix Cov( psi(t) | y(1:t) )
%   logLik - Log-likelihood of innovation log f( y(t) | y(1:t-1) )
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'innovCov','kalmanGain','obsClip','stateClip','eigShift'};
defaultValues = {[],[],1e-3,10,[]};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
innovCov = Opt.innovCov;
kalmanGain = Opt.kalmanGain;
obsClip = Opt.obsClip;
stateClip = Opt.stateClip;
eigShift = Opt.eigShift;

% Shift 0 and 1 observations away from boundaries
p = length(y);
y(y==0) = obsClip;
y(y==1) = 1-obsClip;

logistic = @(x) 1./(1+exp(-x));	% Logistic function
logisticDeriv = @(x) exp(-x)./(1+exp(-x)).^2; % Derivative of logistic function
e = y - logistic(psiPred); % Error (innovation)
% Diagonal Jacobian matrix (using spdiags for quicker multiplication)
jacobian = spdiags(logisticDeriv(psiPred),0,p,p);

% Innovation covariance
if isempty(innovCov)
    innovCov = jacobian*psiPredCov*jacobian' + obsCov;
%     minEig = min(eig(S));
%     if minEig < eigShift
%         shiftAmt = max(2*abs(minEig),eigShift);
%         S = S + shiftAmt*eye(p);
%     end
end
logLik = mvnpdf(e,0,innovCov);

% Near-optimal Kalman gain matrix
if isempty(kalmanGain)
    kalmanGain = psiPredCov*jacobian'/innovCov;
end

% Updated state estimate and covariance matrix
psi = psiPred + kalmanGain*e;
psiCov = (eye(p) - kalmanGain*jacobian)*psiPredCov;

% Ensure that symmetry of the covariance matrix is retained (to account for
% numerical errors)
psiCov = (psiCov+psiCov')/2;

% Ensure that positive definiteness of the covariance matrix is retained.
% If it is not, then add a small diagonal matrix to shift the eigenvalues.
if ~isempty(eigShift)
    minEig = min(eig(psiCov));
    if minEig < eigShift
        shiftAmt = max(2*abs(minEig),eigShift);
        psiCov = psiCov + shiftAmt*eye(p);
    end
end

% Clip state estimates that deviate too far from 0
psi(psi < -stateClip) = -stateClip;
psi(psi > stateClip) = stateClip;

end

