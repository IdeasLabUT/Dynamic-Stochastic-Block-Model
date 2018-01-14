function [psiPred,psiPredCov,Opt] = ekfPredict(psi,psiCov,stateTrans, ...
    transCov,Opt)
%ekfPredict Predict phase of extended Kalman filter for dynamic SBM
%   [psiPred,psiPredCov,Opt] = ekfPredict(psi,psiCov,stateTrans,transCov,Opt)
%
%   Inputs:
%   psi - State estimate at previous time E[ psi(t-1) | y(1:t-1) ]
%   psiCov - Covariance matrix of previous state estimate 
%       Cov( psi(t-1) | y(1:t-1) )
%   stateTrans - State transition matrix applied to previous state at time
%                t-1
%   transCov - Process noise covariance matrix driving state dynamics
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional parameters (specified as fields of Opt [default in brackets]):
%   'stateClip' - Value at which to clip state estimates. Estimates that
%                 exceed stateClip in magnitude will be clipped. [ 10 ]
%   'eigShift' - Amount to shift eigenvalues of estimated state covariance
%                if positive definiteness is lost. Set to empty matrix to
%                ignore check for positive definiteness. [ [] ]
%
%   Outputs:
%   psiPred - Predicted state at current time E[ psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state Cov( psi(t) | y(1:t-1) )
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'stateClip','eigShift'};
defaultValues = {10,[]};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
stateClip = Opt.stateClip;
eigShift = Opt.eigShift;

% Apply dynamic model to previous state estimate to obtain predicted state
psiPred = stateTrans*psi;
psiPredCov = stateTrans*psiCov*stateTrans' + transCov;

% Ensure that symmetry of the covariance matrix is retained (to account for
% numerical errors)
psiPredCov = (psiPredCov+psiPredCov')/2;

% Ensure that positive definiteness of the covariance matrix is retained.
% If it is not, then add a small diagonal matrix to shift the eigenvalues.
if ~isempty(eigShift)
    minEig = min(eig(psiPredCov));
    if minEig < eigShift
        shiftAmt = max(2*abs(minEig),eigShift);
        psiPredCov = psiPredCov + shiftAmt*eye(p);
    end
end

% Clip state predictions that deviate too far from 0
psiPred(psiPred < -stateClip) = -stateClip;
psiPred(psiPred > stateClip) = stateClip;

end

