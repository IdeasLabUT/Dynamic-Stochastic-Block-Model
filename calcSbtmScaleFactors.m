function scaleFactors = calcSbtmScaleFactors(adjPrev,class,classPrev, ...
    probNew,probExist,probMargPrev,directed)
%calcSbtmScaleFactors Calculate matrix of scaling factors for the SBTM
%   scaleFactors = calcSbtmScaleFactors(adjPrev,class,classPrev, ...
%       probNew,probExist,probMargPrev,directed)
%
%   calcSbtmScaleFactors calculates the matrix of scaling factors for the
%   stochastic block transition model (SBTM). These scaling factors are
%   used to scale the observed edges so that they have the same conditional
%   mean, which allows their sample mean to be used to estimate the
%   probabilities of forming new edges and of existing edges re-occurring.
%
%   Inputs:
%   adjPrev - n x n observed adjacency matrix at previous time step where
%             rows and columns correspond to active nodes at the current
%             time step
%   class - Length n vector containing class membership of each node at the
%           current time step (scalar from 1 to the number of blocks k)
%   classPrev - Length n vector containing the class membership of each
%               current node at the previous time step. If a node was not
%               present at the previous time step, its entry should be set
%               to 0.
%   probNew - k x k matrix of probabilities of forming a new edge in each
%             block
%   probExist - k x k matrix of probabilities of an existing edge
%               re-occurring in each block
%   probMargPrev - k x k matrix of marginal edge probabilities for each
%                  block at previous time step
%   directed - Whether the graph is directed (set to true or false,
%              defaults to false)

% Author: Kevin S. Xu

if nargin < 7
    directed = false;
end

n = length(class);
k = max(max(class),max(classPrev));

% Set to all ones rather than all zeros to avoid dividing by zero (since
% the adjacency matrix is divided by the scale factor matrix)
scaleFactors = ones(n);

% Current marginal probabilities
probMarg = probNew.*(1-probMargPrev) + probExist.*probMargPrev;

% Calculate scale factors for new nodes

% Mask indicating new nodes
newMask = false(n,n);
newMask(classPrev==0,:) = true;
newMask(:,classPrev==0) = true;
for a = 1:k
    if directed == true
        bStart = 1;
    else
        bStart = a;
    end
    
    for b = bStart:k
        blockMask = false(n,n);
        blockMask(class==a,class==b) = true;
        if directed == false
            blockMask(class==b,class==a) = true;
        end
        newBlock = newMask & blockMask;
        
        scaleFactorNew = 1 - (1-probExist(a,b)/probNew(a,b)) ...
            * probMargPrev(a,b);
        scaleFactors(newBlock) = scaleFactorNew;
    end
end

% Calculate scale factors for nodes present at both time steps

% Binary masks denoting edges and non-edges at previous time
zeroMask = (adjPrev==0);
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
        blockMask(class==a,class==b) = true;
        if directed == false
            blockMask(class==b,class==a) = true;
        end
        
        for aPrev = 1:k
            if directed == true
                bPrevStart = 1;
            else
                bPrevStart = a;
            end
    
            for bPrev = bPrevStart:k
                blockPrevMask = false(n,n);
                blockPrevMask(classPrev==aPrev,classPrev==bPrev) = true;
                if directed == false
                    blockPrevMask(classPrev==bPrev,classPrev==aPrev) = true;
                end
                blockBothMask = blockMask & blockPrevMask;
                
                % If no pairs belong currently to classes (a,b) and
                % previously to classes (aPrev,bPrev), move onto the next
                % tuple of classes
                if sum(blockBothMask) == 0
                    continue
                end
                
                % For nodes that didn't change classes, leave scale factor
                % at 1
                if (a == aPrev) && (b == bPrev)
                    continue
                end
                
                zeroBlock = zeroMask & blockBothMask;
                oneBlock = oneMask & blockBothMask;
                
                % Upper and lower bounds for scale factor for new edge
                lBound = max(0, (probMarg(a,b)-probMargPrev(aPrev, ...
                    bPrev))/(probNew(a,b)*(1-probMargPrev(aPrev,bPrev))));
                uBound = min(1/probNew(a,b), probMarg(a,b) ...
                    / (probNew(a,b)*(1-probMargPrev(aPrev,bPrev))));

                % Upper and lower bounds for nodes that stayed in the
                % same class (used for scaling other nodes' scale
                % factors)
                lBoundSame = max(0, (probMarg(a,b)-probMargPrev(a,b)) ...
                    / (probNew(a,b)*(1-probMargPrev(a,b))));
                uBoundSame = min(1/probNew(a,b), probMarg(a,b) ...
                    / (probNew(a,b)*(1-probMargPrev(a,b))));

                scaleFactorNew = lBound + (1 - lBoundSame)/(uBoundSame ...
                    - lBoundSame)*(uBound - lBound);
                % Substitute scale factor for new edge into equation of a
                % line to find scale factor for existing edge
                slope = probNew(a,b)*(probMargPrev(aPrev,bPrev) - 1) ...
                    / (probExist(a,b)*probMargPrev(aPrev,bPrev));
                intercept = probMarg(a,b) / (probExist(a,b) ...
                    * probMargPrev(aPrev,bPrev));
                scaleFactorExist = slope*scaleFactorNew + intercept;

                scaleFactors(zeroBlock) = scaleFactorNew;
                scaleFactors(oneBlock) = scaleFactorExist;
            end
        end
    end
end

end

