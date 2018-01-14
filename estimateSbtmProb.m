function [psi,psiCov,logPost,newExistDens,scaleFactors,Opt] ...
    = estimateSbtmProb(adj,adjPrev,class,classPrev,psiPred,psiPredCov, ...
    probMargPrev,obsCov,scaleFactors,node,Opt)
%estimateSbtmProb Estimate edge probabilities for SBTM
%   [psi,psiCov,logPost,newExistDens,Opt] = estimateSbtmProb(adj, ...
%       adjPrev,class,classPrev,psiPred,psiPredCov,probMargPrev,obsCov,Opt)
%
%   Inputs:
%   adj - Graph adjacency matrix at current time (binary with no self-edges;
%         can be directed, i.e. adj(i,j) = 1 denotes edge from i to j, and
%         adj(i,j) = 0 denotes absence of edge from i to j)
%   adjPrev - Graph adjacency matrix at previous time
%   class - Class membership vector at current time
%   classPrev - Class membership vector at previous time
%   psiPred - Predicted state estimate given observations up to time t-1
%             E[ psi(t) | y(1:t-1) ]
%   psiPredCov - Covariance matrix of predicted state estimate
%                Cov( psi(t) | y(1:t-1) ) 
%   probMargPrev - Marginal block edge probability estimates at the previous
%                  time step (used to estimate SBTM scale factors)
%   obsCov - Covariance matrix of observation noise (set to [] if unknown)
%   scaleFactors - Matrix of SBTM scaling factors. Scaling factors can be
%                  estimated for the entire adjacency matrix by setting
%                  scaleFactors to []. Scaling factors can be estimated
%                  just for entries involving a specific node, i.e. for a
%                  single row and column of the adjacency matrix; in this
%                  case, scaleFactors must be specified along with the node
%                  of interest.
%   node - Node of interest. If estimating scaling factors just for entries
%          involving a single node, i.e. scaleFactors is non-empty, then
%          this input is used to identify which row and column of
%          scaleFactors to update. If scaleFactors is empty, then this
%          input is not used.
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'scaleClip' - Value at which to clip scaled edge forming probability
%                 away from boundaries of 0 and 1 for which the
%                 log-likelihood cannot be calculated. All entries smaller
%                 than scaleClip or larger than 1-scaleClip are clipped.
%                 [ 1e-6 ]
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%
%   Outputs:
%   psi - Updated state estimate E[ psi(t) | y(1:t) ]
%   psiCov - Updated state covariance matrix Cov( psi(t) | y(1:t) )
%   logPost - Log of posterior probability log f( psi(t) | adj(t) )
%   newExistDens - Matrix of sample means of scaled adjacency matrix blocks,
%                  where each column denotes observations at time t
%   scaleFactors - Updated matrix of SBTM scaling factors
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'scaleClip','directed'};
defaultValues = {1e-6,false};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
scaleClip = Opt.scaleClip;
directed = Opt.directed;

n = size(adj,1);
p = length(psiPred);

% logit = @(x) log(x) - log(1-x);
logistic = @(x) 1./(1+exp(-x));

% Compute scale factors and block statistics
thetaNewPredMat = blockvec2mat(logistic(psiPred(1:p/2)),directed);
thetaExistPredMat = blockvec2mat(logistic(psiPred(p/2+1:end)),directed);
probMargPrevMat = blockvec2mat(probMargPrev,directed);
if isempty(scaleFactors)
    % Estimate entire matrix of scaling factors
    scaleFactors = calcSbtmScaleFactors(adjPrev,class,classPrev, ...
        thetaNewPredMat,thetaExistPredMat,probMargPrevMat,directed);
else
    % Estimate only row and column of scaling factor matrix corresponding
    % to node of interest
    scaleFactors = updateSbtmScaleFactorsNode(adjPrev,class,classPrev, ...
        thetaNewPredMat,thetaExistPredMat,probMargPrevMat,scaleFactors, ...
        node,directed);
end
[newDens,existDens,~,~,nNewPairs,nExistPairs,Opt] = calcNewExistDens(cat ...
    (3,adjPrev,adj),cat(3,classPrev,class),cat(3,ones(n),scaleFactors),Opt);
newExistDens = [newDens(:,2); existDens(:,2)];
nNewPairs = nNewPairs(:,2);
nExistPairs = nExistPairs(:,2);

% If necessary, estimate observation noise covariance matrix Sigma
% using psiPred
if isempty(obsCov)
    nPairs = [nNewPairs; nExistPairs];
    nPairs(nPairs==0) = 1;
    obsCov = diag(logistic(psiPred).*(1-logistic(psiPred)) ./ nPairs);
end

% Run EKF to obtain estimate of psi(t)
[psi,psiCov,~,Opt] = ekfUpdate(newExistDens,psiPred,psiPredCov,obsCov,Opt);
thetaNewMat = blockvec2mat(logistic(psi(1:p/2)),directed);
thetaExistMat = blockvec2mat(logistic(psi(p/2+1:end)),directed);

% Calculate log of prior probability f( psi(t) | y(1:t-1) )
logPrior = log(mvnpdf(psi,psiPred,psiPredCov));

% Calculate log-likelihood f( adj(t) | adj(t-1), psi(t) ) by summing over
% all blocks
logLik = 0;
k = size(thetaNewMat,1);
% Binary masks denoting edges and non-edges at previous time
zeroMask = (adjPrev==0);
oneMask = ~zeroMask;
% Remove diagonal from set of non-edges since we are not allowing
% self-edges
zeroMask(diag(true(n,1))) = false;
for c1 = 1:k
    for c2 = 1:k
        blockMask = false(n,n);
        blockMask(class==c1,class==c2) = true;
        zeroBlock = zeroMask & blockMask;
        oneBlock = oneMask & blockMask;
        scaledThetaNew = scaleFactors(zeroBlock) * thetaNewMat(c1,c2);
        scaledThetaExist = scaleFactors(oneBlock) * thetaExistMat(c1,c2);
        
        % Ensure scaled probabilities remain within (0,1) so that
        % logarithms can be taken
        if sum(scaledThetaNew < scaleClip) > 0
            scaledThetaNew(scaledThetaNew < scaleClip) = scaleClip;
        elseif sum(scaledThetaNew > 1-scaleClip)
            scaledThetaNew(scaledThetaNew > 1-scaleClip) = 1-scaleClip;
        end
        if sum(scaledThetaExist < scaleClip)
            scaledThetaExist(scaledThetaExist < scaleClip) = scaleClip;
        elseif sum(scaledThetaExist > 1-scaleClip)
            scaledThetaExist(scaledThetaExist > 1-scaleClip) = 1-scaleClip;
        end
        
        % Add contribution of this block to the log-likelihood
        logLik = logLik + sum(adj(zeroBlock).*log(scaledThetaNew) ...
            + (1-adj(zeroBlock)).*log(1-scaledThetaNew)) ...
            + sum(adj(oneBlock).*log(scaledThetaExist) + (1-adj(oneBlock)) ...
            .*log(1-scaledThetaExist));
    end
end

% Calculate log of posterior probability f( psi(t) | adj(t) )
logPost = logPrior + logLik;

end

