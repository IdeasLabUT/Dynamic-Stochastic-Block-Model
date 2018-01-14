% Fit dynamic stochastic block model (SBM) in both a priori and a posteriori
% settings to data generated from an SBM with parameters evolving according
% to a linear model with logistic transformation.
% Author: Kevin S. Xu

nRuns = 5;  % Number of simulation runs
tMax = 10;  % Number of time steps
n = 128;    % Number of nodes
k = 4;  % Number of classes
directed = false;   % Whether snapshots are directed or undirected
% Number of states
if directed == true
    p = k^2;
else
    p = k*(k+1)/2;
end

% Mean of initial block probabilities
initProb = 2*0.0417*ones(k);
initProb(diag(true(k,1))) = 2*0.1290;
% Covariance matrix of initial state
initStateCov = 0.04*eye(p);

% Mean of process noise driving state evolution
stateMean = zeros(p,1);
% Covariance matrix of process noise
stateCovIn = 0.01;
stateCovOut = 0.0025;
% Probability of changing class at each time step
changeProb = 0.1;

% Optional parameters
Opt.directed = directed;
% Number of random initializations for k-means step of spectral clustering
Opt.nKmeansReps = 5;
Opt.maxIter = 200;  % Maximum number of local search iterations
Opt.output = 1; % Level of output to display in console
Opt.model = 1:tMax; % Index vector for model at each time step

%% Initialize variables storing data from all simulation runs
msePriEkfRuns = zeros(tMax,nRuns);
msePriStaticRuns = zeros(tMax,nRuns);
msePostEkfRuns = zeros(tMax,nRuns);
msePostStaticRuns = zeros(tMax,nRuns);
randPostEkfRuns = zeros(tMax,nRuns);
randPostStaticRuns = zeros(tMax,nRuns);

%% Start simulation
logistic = @(x) 1./(1+exp(-x));    % Logistic function
logit = @(x) log(x)-log(1-x);   % Logit function (inverse of logistic)
initState = logit(blockmat2vec(initProb,directed));
stateCov = generateStateCov(k,stateCovIn,stateCovOut,directed);
output = Opt.output;

for run = 1:nRuns
    disp(['Run number ' int2str(run)])
    
    %% Generate data and calculate a priori block densities
    psi = zeros(p,tMax);
    class = zeros(n,tMax);
    
    class(:,1) = reshape(repmat(1:k,n/k,1),n,1);
    
    % Generate state sequence by Gaussian random walk
    psi(:,1) = mvnrnd(initState,initStateCov)' + mvnrnd(stateMean, ...
        stateCov)';
    for t = 2:tMax
        psi(:,t) = psi(:,t-1) + mvnrnd(stateMean,stateCov)';
        
        % Randomly select some nodes to change class
        class(:,t) = class(:,t-1);
        changeNodes = randsample(n,ceil(changeProb*n));
        newClassIndices = randi(k-1,1,length(changeNodes));
        for idx = 1:length(changeNodes)
            currClasses = class(changeNodes(idx),t);
            newClasses = setdiff(1:k,currClasses);
            class(changeNodes(idx),t) = newClasses(newClassIndices(idx));
        end
    end
    % Transform states into probabilities
    theta = logistic(psi);
    % Generate sequence of adjacency matrices
    adj = generateDsbm(class,blockvec2mat(theta,directed),directed);
    
    % Calculate densities of a priori blocks
    [blockDens,nEdges,nPairs,OptPriEkf] = calcBlockDens(adj,class,Opt);
    % Compute observation noise variances
    noiseVar = theta.*(1-theta)./nPairs;
    noiseVar(nPairs==0) = 0; % Avoid division by 0 for empty group
    obsCov = zeros(p,p,tMax);
    for t = 1:tMax
        obsCov(:,:,t) = diag(noiseVar(:,t));
    end
    
    %% Estimate states using EKF with a priori classes
    if output > 0
        disp('Estimating states using EKF with a priori classes')
    end
    stateTrans = repmat(eye(p),[1 1 tMax]);
    stateCovArray = repmat(stateCov,[1 1 tMax]);
    [psiPriEkf,~,~,OptPriEkf] = ekfDsbm(blockDens,stateTrans,stateCovArray, ...
        obsCov,initState,initStateCov,OptPriEkf);
    thetaPriEkf = logistic(psiPriEkf);
    
    %% Estimate states using EKF with classes estimated by local search
    if output > 0
        disp('Estimating states using EKF with a posteriori class estimates')
    end
    [classPostEkf,psiPostEkf,~,~,OptPostEkf] = ekfDsbmLocalSearch(adj,k, ...
        stateTrans,stateCovArray,obsCov,initState,initStateCov,Opt);
    
    %% Estimate states using static spectral clustering at each time step
    classPostStatic = zeros(n,tMax);
    thetaPostStatic = zeros(p,tMax);
    for t = 1:tMax
        [classPostStatic(:,t),~,~,OptPostStatic] = spectralClusterSbm ...
            (adj(:,:,t),k,Opt);
        [thetaPostStatic(:,t),~,OptPostStatic] = estimateSbmProb ...
            (adj(:,:,t),classPostStatic(:,t),OptPostStatic);
    end
    
    %% Permute estimated classes for maximum agreement with true classes     
    [classPostEkf,bestPerm] = permuteClasses(class,classPostEkf);
    psiPostEkfMat = blockvec2mat(psiPostEkf,directed);
    psiPostEkf = blockmat2vec(psiPostEkfMat(bestPerm,bestPerm,:), ...
        directed);
    thetaPostEkf = logistic(psiPostEkf);
    
    [classPostStatic,bestPerm] = permuteClasses(class,classPostStatic);
    thetaPostStaticMat = blockvec2mat(thetaPostStatic,directed);
    thetaPostStatic = blockmat2vec(thetaPostStaticMat(bestPerm,bestPerm,:), ...
        directed);
    
    %% Accuracy measures
    
	% Mean-squared estimation error of EKF with a priori classes
	msePriEkf = sum((theta-thetaPriEkf).^2,1);
    % Mean-squared estimation error of static SBM with a priori classes
    msePriStatic = sum((theta-blockDens).^2,1);
	% Mean-squared estimation error of EKF with estimated classes
	msePostEkf = sum((theta-thetaPostEkf).^2,1);
	% Mean-squared estimation error of static SBM with estimated classes
	msePostStatic = sum((theta-thetaPostStatic).^2,1);
    
    % Adjusted Rand index of estimated classes
    randPostEkf = zeros(1,tMax);
    randPostStatic = zeros(1,tMax);
    for t = 1:tMax
        randPostEkf(t) = valid_RandIndex(class(:,t),classPostEkf(:,t));
        randPostStatic(t) = valid_RandIndex(class(:,t),classPostStatic(:,t));
    end
    
    %% Store variables from this run as slice of script variables
    msePriEkfRuns(:,run) = msePriEkf;
    msePriStaticRuns(:,run) = msePriStatic;
    msePostEkfRuns(:,run) = msePostEkf;
    msePostStaticRuns(:,run) = msePostStatic;
    randPostEkfRuns(:,run) = randPostEkf;
    randPostStaticRuns(:,run) = randPostStatic;
end

%% Display summary statistics
disp('-----')
disp('Summary statistics:')
disp(['A priori EKF MSE: ' num2str(mean(msePriEkfRuns(:))) ...
    ' +/- ' num2str(std(mean(msePriEkfRuns,1))/sqrt(nRuns))])
disp(['A priori static SBM MSE: ' num2str(mean(msePriStaticRuns(:))) ...
    ' +/- ' num2str(std(mean(msePriStaticRuns,1))/sqrt(nRuns))])
disp(['A posteriori EKF MSE: ' num2str(mean(msePostEkfRuns(:))) ...
    ' +/- ' num2str(std(mean(msePostEkfRuns,1))/sqrt(nRuns))])
disp(['A posteriori static SBM MSE: ' num2str(mean(msePostStaticRuns(:))) ...
    ' +/- ' num2str(std(mean(msePostStaticRuns,1))/sqrt(nRuns))])
disp(['A posteriori EKF adjusted Rand index: ' num2str(mean ...
    (randPostEkfRuns(:))) ' +/- ' num2str(std(mean(randPostEkfRuns, ...
    1))/sqrt(nRuns))])
disp(['A posteriori static SBM adjusted Rand index: ' num2str(mean ...
    (randPostStaticRuns(:))) ' +/- ' num2str(std(mean(randPostStaticRuns, ...
    1))/sqrt(nRuns))])
