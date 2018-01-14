% Fit dynamic stochastic block model (SBM) to Enron email data
% Author: Kevin S. Xu

deltaT = 7;	% Length of each time step (in days)
thres = 1;  % Threshold for edges (number of emails required to place edge)
includeCc = true;
directed = true;

% Parameters for with known classes
kPri = 7;
pPriDsbm = kPri^2;
initCovPriDsbm = eye(pPriDsbm);
stateCovInPriDsbm = 0.01;
stateCovOutPriDsbm = 0;

pPriSbtm = 2*pPriDsbm;
initCovPriSbtm = eye(pPriSbtm);
stateCovInPriSbtm = 0.01;
stateCovOutPriSbtm = 0;

% Parameters for unknown classes
kPost = 4;
pPostDsbm = kPost^2;
initCovPostDsbm = eye(pPostDsbm);
stateCovInPostDsbm = 0.1;
stateCovOutPostDsbm = 0.02;

pPostSbtm = 2*pPostDsbm;
initCovPostSbtm = eye(pPostSbtm);
stateCovInPostSbtm = 0.1;
stateCovOutPostSbtm = 0.02;

% Optional parameters
Opt.directed = directed;
% Number of random initializations for k-means step of spectral clustering
Opt.nKmeansReps = 5;
Opt.maxIter = 200;  % Maximum number of local search iterations
Opt.output = 1; % Level of output to display in console

nSynNets = 10;

%% Load data
if directed == false
	str1 = 'Undirected';
else
	str1 = 'Directed';
end
if includeCc == false
	str2 = 'NoCc';
else
	str2 = 'WithCc';
end
load(['Enron' str1 str2 '_' int2str(deltaT) 'Days.mat'])

% Truncate weeks at beginning and end with low participation
if deltaT == 7
	adj = adj(:,:,57:176);
	traceStartDate = traceStartDate + 56*deltaT;
	traceEndDate = traceEndDate - 13*deltaT;
end

%% Pre-processing
[n,~,tMax] = size(adj);

% Dichotomize edges
adj(adj<thres) = 0;
adj(adj>=thres) = 1;

% Combine roles into 7 groups:
% 1: Director, Managing Director
% 2: CEO
% 3: President
% 4: Vice President
% 5: Manager
% 6: Trader
% 7: Employee, In House Lawyer, and N/A
rolesOld = roles;
roles = kPri*ones(n,1);    % Set default role to number of classes
roles(rolesOld == find(strcmp(roleLabels,'Director'),1)) = 1;
roles(rolesOld == find(strcmp(roleLabels,'Managing Director'),1)) = 1;
roles(rolesOld == find(strcmp(roleLabels,'CEO'),1)) = 2;
roles(rolesOld == find(strcmp(roleLabels,'President'),1)) = 3;
roles(rolesOld == find(strcmp(roleLabels,'Vice President'),1)) = 4;
roles(rolesOld == find(strcmp(roleLabels,'Manager'),1)) = 5;
roles(rolesOld == find(strcmp(roleLabels,'Trader'),1)) = 6;

logistic = @(x) 1./(1+exp(-x));	% Logistic function

%% Estimate states using EKF with a priori classes
disp('Estimating states using EKF with a priori classes')

Opt.nClasses = kPri;

% Fit a priori HM-SBM
stateTransPriDsbm = eye(pPriDsbm);
stateCovPriDsbm = generateStateCov(kPri,stateCovInPriDsbm, ...
    stateCovOutPriDsbm,directed);

[blockDens,nEdges,nPairs,OptPriDsbm] = calcBlockDens(adj,roles,Opt);
[psiPriDsbm,psiCovPriDsbm,~,OptPriDsbm] = ekfDsbm(blockDens,stateTransPriDsbm, ...
    stateCovPriDsbm,[],[],initCovPriDsbm,OptPriDsbm);
thetaPriDsbmMat = blockvec2mat(logistic(psiPriDsbm),directed);

% Compute 95% confidence intervals for class connection probabilities
% estimated by a priori EKF
psiStdPriDsbm = zeros(pPriDsbm,tMax);
for t = 1:tMax
    psiStdPriDsbm(:,t) = sqrt(diag(psiCovPriDsbm(:,:,t)));
end
thetaPriDsbmUpperMat = blockvec2mat(logistic(psiPriDsbm + 2*psiStdPriDsbm), ...
    directed);
thetaPriDsbmLowerMat = blockvec2mat(logistic(psiPriDsbm - 2*psiStdPriDsbm), ...
    directed);

% Fit a priori SBTM
stateTransPriSbtm = eye(pPriSbtm);
stateCovPriSbtm = stateCovOutPriSbtm*ones(pPriSbtm);
stateCovPriSbtm(diag(true(pPriSbtm,1))) = stateCovInPriSbtm;

[psiPriSbtm,psiCovPriSbtm,~,~,~,thetaPriSbtm,OptPriSbtm] = ekfSbtm(adj, ...
    repmat(roles,1,tMax),stateTransPriSbtm,stateCovPriSbtm,[],[], ...
    initCovPriSbtm,Opt);
probNewPri = blockvec2mat(logistic(psiPriSbtm(1:pPriSbtm/2,:)),directed);
probExistPri = blockvec2mat(logistic(psiPriSbtm(pPriSbtm/2+1:end,:)),directed);
thetaPriSbtmMat = blockvec2mat(thetaPriSbtm,directed);

%% Estimate states using EKF with classes estimated by local search
disp('Estimating states using EKF with a posteriori class estimates')

stateTransPostDsbm = eye(pPostDsbm);
stateCovPostDsbm = generateStateCov(kPost,stateCovInPostDsbm, ...
    stateCovOutPostDsbm,directed);

% Fit a posteriori HM-SBM
OptPostEkfDsbm = Opt;
OptPostEkfDsbm.nClasses = kPost;
[classPostEkfDsbm,psiPostEkfDsbm,psiCovPostEkfDsbm,~,OptPostEkfDsbm] ...
    = ekfDsbmLocalSearch(adj,kPost,stateTransPostDsbm,stateCovPostDsbm, ...
    [],[],initCovPostDsbm,OptPostEkfDsbm);
thetaPostEkf = logistic(psiPostEkfDsbm);

% Fit a posteriori SBTM
stateTransPostSbtm = eye(pPostSbtm);
stateCovPostSbtm = stateCovOutPostSbtm*ones(pPostSbtm);
stateCovPostSbtm(diag(true(pPostSbtm,1))) = stateCovInPostSbtm;

OptPostEkfSbtm = Opt;
OptPostEkfSbtm.nClasses = kPost;
OptPostEkfSbtm.output = 2;
[classPostEkfSbtm,psiPostEkfSbtm,~,~,~,thetaPostSbtm,OptPostEkfSbtm] ...
    = ekfSbtmLocalSearch(adj,kPost,stateTransPostSbtm,stateCovPostSbtm, ...
    [],[],initCovPostSbtm,OptPostEkfSbtm);
probNewPost = blockvec2mat(logistic(psiPostEkfSbtm(1:pPostSbtm/2,:)),directed);
probExistPost = blockvec2mat(logistic(psiPostEkfSbtm(pPostSbtm/2+1:end,:)), ...
    directed);
thetaPostSbtmMat = blockvec2mat(thetaPostSbtm,directed);

%% Calculate forecast error of block densities
psiPredPriEkf = stateTransPriDsbm*psiPriDsbm;
thetaPredPriEkf = [zeros(pPriDsbm,1) logistic(psiPredPriEkf(:,1:tMax-1))];
msePredPriEkf = sum((thetaPredPriEkf(:,2:tMax) - blockDens(:,2:tMax)).^2);

blockDensPost = calcBlockDens(adj,classPostEkfDsbm,OptPostEkfDsbm);
psiPredPostEkf = stateTransPostDsbm*psiPostEkfDsbm;
thetaPredPostEkf = [zeros(pPostDsbm,1) logistic(psiPredPostEkf(:,1:tMax-1))];
msePredPostEkf = sum((thetaPredPostEkf(:,2:tMax) - blockDensPost(:,2:tMax)).^2);

%% Calculate forecast error of links
disp('Calculating ROC of link forecast')

% Exponentially-weighted moving average (EWMA) predictor
predMatEwma = zeros(n,n,tMax);
ff = 0.5;   % Forgetting factor for EWMA
ccWt = 0.01;   % Weight of EKF predictor in convex combination

rolesMat = repmat(roles,1,tMax);
% Do not assign role before node first appears
nodeActive = isNodeActive(adj);
rolesMat(cumsum(nodeActive,2)==0) = 0;
% Link forecast based on a priori classes (employee roles)
predMatPriEkfDsbm = predAdjMatDsbm(adj,blockvec2mat ...
    (thetaPredPriEkf,directed),rolesMat);
predMatPriEkfSbtm = predAdjMatSbtm(adj,probNewPri,probExistPri,thetaPriSbtmMat ...
    (:,:,1),rolesMat,Opt);
% Link forecast based on a posteriori class estimates
predMatPostEkfDsbm = predAdjMatDsbm(adj,blockvec2mat(thetaPredPostEkf, ...
    directed),classPostEkfDsbm);
predMatPostEkfSbtm = predAdjMatSbtm(adj,probNewPost,probExistPost, ...
    thetaPostSbtmMat,classPostEkfSbtm,Opt);

predMatEwma(:,:,2) = adj(:,:,1);
for t = 3:tMax
    predMatEwma(:,:,t) = ff*predMatEwma(:,:,t-1) + (1-ff)*adj(:,:,t-1);
end
predMatPriEkfDsbmEwma = ccWt*predMatPriEkfDsbm + (1-ccWt)*predMatEwma;
predMatPriEkfSbtmEwma = ccWt*predMatPriEkfSbtm + (1-ccWt)*predMatEwma;
predMatPostEkfDsbmEwma = ccWt*predMatPostEkfDsbm + (1-ccWt)*predMatEwma;
predMatPostEkfSbtmEwma = ccWt*predMatPostEkfSbtm + (1-ccWt)*predMatEwma;

nTotPairs = n*(n-1)*(tMax-1);
if directed == false
    nTotPairs = nTotPairs/2;
end

adjVect = reshape(adjmat2vec(adj(:,:,2:end),directed),1,nTotPairs);
[fprPriDsbm,tprPriDsbm,~,aucPriDsbm] = perfcurve(adjVect,reshape(adjmat2vec ...
    (predMatPriEkfDsbm(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPriSbtm,tprPriSbtm,~,aucPriSbtm] = perfcurve(adjVect,reshape(adjmat2vec ...
    (predMatPriEkfSbtm(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPostDsbm,tprPostDsbm,~,aucPostDsbm] = perfcurve(adjVect,reshape(adjmat2vec ...
    (predMatPostEkfDsbm(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPostSbtm,tprPostSbtm,~,aucPostSbtm] = perfcurve(adjVect,reshape(adjmat2vec ...
    (predMatPostEkfSbtm(:,:,2:tMax),directed),1,nTotPairs),1);
[fprEwma,tprEwma,~,aucEwma] = perfcurve(adjVect,reshape(adjmat2vec ...
    (predMatEwma(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPriDsbmEwma,tprPriDsbmEwma,~,aucPriDsbmEwma] = perfcurve(adjVect, ...
    reshape(adjmat2vec(predMatPriEkfDsbmEwma(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPriSbtmEwma,tprPriSbtmEwma,~,aucPriSbtmEwma] = perfcurve(adjVect, ...
    reshape(adjmat2vec(predMatPriEkfSbtmEwma(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPostDsbmEwma,tprPostDsbmEwma,~,aucPostDsbmEwma] = perfcurve(adjVect,reshape ...
    (adjmat2vec(predMatPostEkfDsbmEwma(:,:,2:tMax),directed),1,nTotPairs),1);
[fprPostSbtmEwma,tprPostSbtmEwma,~,aucPostSbtmEwma] = perfcurve(adjVect,reshape ...
    (adjmat2vec(predMatPostEkfSbtmEwma(:,:,2:tMax),directed),1,nTotPairs),1);

%% Check how well edge durations and intervals are reproduced
disp('Calculating edge durations and intervals')
[durObs,intObs] = calcEdgeDurationsIntervals(adj,Opt);

durSbtm = [];
intSbtm = [];
durDsbm = [];
intDsbm = [];
probInit = thetaPostSbtmMat(:,:,1);
thetaPostEkfMat = blockvec2mat(thetaPostEkf,directed);
classPostSbtmActive = classPostEkfSbtm;
classPostSbtmActive(isNodeActive(adj) == false) = 0;
classPostDsbmActive = classPostEkfDsbm;
classPostDsbmActive(isNodeActive(adj) == false) = 0;
parfor iSyn = 1:nSynNets
    disp(['Generating synthetic network ' int2str(iSyn)])
    
    adjSbtm = generateSbtm(classPostSbtmActive,probInit,probNewPost,probExistPost, ...
        directed);
    [durSbtmNet,intSbtmNet] = calcEdgeDurationsIntervals(adjSbtm,Opt);
    durSbtm = [durSbtm; durSbtmNet]; %#ok<*AGROW>
    intSbtm = [intSbtm; intSbtmNet];
    
    adjDsbm = generateDsbm(classPostDsbmActive,thetaPostEkfMat,directed);
    [durDsbmNet,intDsbmNet] = calcEdgeDurationsIntervals(adjDsbm,Opt);
    durDsbm = [durDsbm; durDsbmNet];
    intDsbm = [intDsbm; intDsbmNet];
end

%% Summary statistics
disp('-----')
disp('Summary statistics:')
disp(['Mean a priori EKF block density forecast MSE: ' ...
    num2str(mean(msePredPriEkf),'%5.3f')])
disp(['Mean a posteriori EKF block density forecast MSE: ' ...
    num2str(mean(msePredPostEkf),'%5.3f')])
disp(['A priori EKF DSBM AUC: ' num2str(aucPriDsbm,'%5.3f')])
disp(['A posteriori EKF DSBM AUC: ' num2str(aucPostDsbm,'%5.3f')])
disp(['A priori EKF SBTM AUC: ' num2str(aucPriSbtm,'%5.3f')])
disp(['A posteriori EKF SBTM AUC: ' num2str(aucPostSbtm,'%5.3f')])
disp(['Exponentially-weighted moving average (EWMA) AUC: ' ...
    num2str(aucEwma,'%5.3f')])
disp(['A priori EKF + EWMA DSBM AUC: ' num2str(aucPriDsbmEwma,'%5.3f')])
disp(['A posteriori EKF + EWMA DSBM AUC: ' num2str(aucPostDsbmEwma,'%5.3f')])
disp(['A priori EKF + EWMA SBTM AUC: ' num2str(aucPriSbtmEwma,'%5.3f')])
disp(['A posteriori EKF + EWMA SBTM AUC: ' num2str(aucPostSbtmEwma,'%5.3f')])

%% Plot all HM-SBM estimated probabilities for one block
figure(1)
row = 2;
col = 3;
plot(1:tMax,reshape(thetaPriDsbmMat(row,col,:),1,tMax),'b', ...
    1:tMax,reshape(thetaPriDsbmLowerMat(row,col,:),1,tMax),':b', ...
    1:tMax,reshape(thetaPriDsbmUpperMat(row,col,:),1,tMax),':b')
legend('Estimated edge probability','95% confidence intervals', ...
    'Location','Best')
xlabel('Week')
ylabel('Edge probability')
title(['Estimated edge probability from ' roleLabels{row} 's to ' ...
    roleLabels{col} 's'])

%% Plot heat map of mean estimated HM-SBM edge probabilities
figure(2)
imagesc(mean(thetaPriDsbmMat,3))
colorbar
colormap gray
xlabel('Recipient class')
ylabel('Sender class')
set(gca,'XTickLabel',roleLabels)
set(gca,'XTickLabelRotation',45)
set(gca,'YTickLabel',roleLabels)

%% Plot ROC curves for link forecasts
figure(3)
h = plot(fprPostSbtm,tprPostSbtm,fprPostDsbm,tprPostDsbm,fprEwma,tprEwma, ...
    fprPostSbtmEwma,tprPostSbtmEwma,fprPostDsbmEwma,tprPostDsbmEwma);
set(h(2),'LineStyle','--')
set(h(3),'LineStyle','-.')
set(h(5),'LineStyle','--')
xlabel('False positive rate')
ylabel('True positive rate')
legend('SBTM','HM-SBM','EWMA','SBTM + EWMA','HM-SBM + EWMA', ...
    'Location','SouthEast')
title('ROC curves for link forecasts')
