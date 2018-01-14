function predMat = predAdjMatSbtm(adj,probNewMat,probExistMat,probInit, ...
    class,Opt)
%predAdjMatSbtm Forecast adjacency matrix at next time step using SBTM
%   predMat = predAdjMatSbtm(adj,probNewMat,probExistMat,probInit, ...
%       class,Opt)
%
%   Inputs:
%   adj - 3-D array of graph adjacency matrices, where each slice along the
%         third dimension denotes the adjacency matrix at time t. Each
%         adjacency matrix is binary with no self-edges and can be directed,
%         i.e. w(i,j,t) = 1 denotes an edge from i to j at time t, and
%         w(i,j,t) = 0 denotes the absence of an edge from i to j at time t.
%   probNewMat - 3-D array where entry (a,b,t) denotes the probability that
%                a non-edge in block (a,b) at time t-1 becomes an edge at
%                time t
%   probExistMat - 3-D array where entry (a,b,t) denotes the probability
%                  that an edge in block (a,b) at time t-1 re-occurs at
%                  time t
%   probInitMat - Matrix of class connection probabilities between classes
%                 at initial time step
%   class - Matrix of class membership vectors, where each column denotes
%           the class membership at time t. Set elements to 0 to indicate
%           that a node is inactive during a time step.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%                [ false ]
%
%   Outputs:
%   predMat - 3-D array of edge probabilities, where entry (i,j,t) denotes
%             the probability of forming an edge from node i to j at time t

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed'};
defaultValues = {false};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;

[n,~,tMax] = size(adj);
k = size(probNewMat,1);

predMat = zeros(n,n,tMax);

% t = 2 case: generate prediction using HM-SBM
predMat(:,:,[1 2]) = predAdjMatDsbm(adj(:,:,[1 2]),cat(3,probInit, ...
    zeros(k,k)),class(:,[1 2]));

% t = 3 and higher: generate prediction using SBTM
for t = 3:tMax
    predMatCurr = zeros(n,n);
    
    % Binary masks denoting edges and non-edges at previous time
    zeroMask = (adj(:,:,t-1)==0);
    oneMask = ~zeroMask;
    % Remove diagonal from set of non-edges since we are not allowing
    % self-edges
    zeroMask(diag(true(n,1))) = false;
    
    for a = 1:k
        if directed == true
            bStart = 1;
        else
            bStart = a;
        end
        
        for b = bStart:k
            blockMask = false(n,n);
            blockMask(class(:,t-1)==a,class(:,t-1)==b) = true;
            if directed == false
                blockMask(class(:,t-1)==b,class(:,t-1)==a) = true;
            end
            
            zeroBlock = zeroMask & blockMask;
            oneBlock = oneMask & blockMask;
            
            % Predicted probability of an edge is either probability of new
            % edge forming or existing edge re-occurring
            predMatCurr(zeroBlock) = probNewMat(a,b,t-1);
            predMatCurr(oneBlock) = probExistMat(a,b,t-1);
        end
    end
    
    predMat(:,:,t) = predMatCurr;
end


end

