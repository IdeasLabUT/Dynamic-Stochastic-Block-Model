function scaleFactors = updateSbtmScaleFactorsNode(adjPrev,class,classPrev, ...
    probNew,probExist,probMargPrev,scaleFactors,node,directed)
%updateSbtmScaleFactorsNode Update SBTM scale factors for a single node
%   scaleFactors = updateSbtmScaleFactorsNode(adjPrev,class,classPrev, ...
%       probNew,probExist,probMargPrev,scaleFactors,node,directed)
%
%   Updates the SBTM scaling factors for all adjacency matrix entries
%   involving a single specified node rather than re-estimating the entire
%   matrix of scaling factors.
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
%   scaleFactors - n x n matrix of current SBTM scaling factor estimates
%   node - Node number for which we are updating the scaling factor
%          estimates
%   directed - Whether the graph is directed (set to true or false,
%              defaults to false)

% Author: Kevin S. Xu

if nargin < 9
    directed = false;
end

k = max(max(class),max(classPrev));

% Current marginal probabilities
probMarg = probNew.*(1-probMargPrev) + probExist.*probMargPrev;

aPrev = classPrev(node);
a = class(node);
if aPrev == 0
    % Node is a new node
    for b = 1:k
        classMask = (class==b);
        if sum(classMask) == 0
            continue
        end
        
        % Edges from node of interest
        scaleFactorNew = 1 - (1-probExist(a,b)/probNew(a,b))*probMargPrev(a,b);
        scaleFactors(node,classMask) = scaleFactorNew;
        
        % Edges to node of interest
        if directed == true
            scaleFactorNew = 1 - (1-probExist(b,a)/probNew(b,a)) ...
                *probMargPrev(b,a);
        end
        scaleFactors(classMask,node) = scaleFactorNew;
    end
else
    % Node was present at previous time step
    
    % Binary masks denoting edges and non-edges from and to node of interest
    % at previous time
    zeroFromMask = (adjPrev(node,:)==0);
    oneFromMask = ~zeroFromMask;
    % Remove self-edge from set of non-edges
    zeroFromMask(node) = false;
    if directed == true
        zeroToMask = (adjPrev(:,node)==0);
        oneToMask = ~zeroToMask;
        zeroToMask(node) = false;
    end
    
    for b = 1:k
        classMask = (class==b);
        if sum(classMask) == 0
            continue
        end
        
        for bPrev = 1:k
            classPrevMask = (classPrev==bPrev);
            classBothMask = classMask & classPrevMask;
            
            % If no pairs of nodes possibly connecting to node of interest
            % belong to class b and previously to class bPrev, move onto
            % next classes
            if sum(classBothMask) == 0
                continue
            end
            
            % For nodes that didn't change classes, set scale factor
            % at 1
            if (a == aPrev) && (b == bPrev)
                scaleFactors(node,classBothMask') = 1;
                scaleFactors(classBothMask,node) = 1;
                continue
            end

            zeroBlockFrom = zeroFromMask & classBothMask';
            oneBlockFrom = oneFromMask & classBothMask';
            
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

            scaleFactors(node,zeroBlockFrom) = scaleFactorNew;
            scaleFactors(node,oneBlockFrom) = scaleFactorExist;
            
            if directed == true
                % Calculate scale factors with classes reversed
                zeroBlockTo = zeroToMask & classBothMask;
                oneBlockTo = oneToMask & classBothMask;
                
                % Upper and lower bounds for scale factor for new edge
                lBound = max(0, (probMarg(b,a)-probMargPrev(bPrev, ...
                    aPrev))/(probNew(b,a)*(1-probMargPrev(bPrev,aPrev))));
                uBound = min(1/probNew(b,a), probMarg(b,a) ...
                    / (probNew(b,a)*(1-probMargPrev(bPrev,aPrev))));

                % Upper and lower bounds for nodes that stayed in the
                % same class (used for scaling other nodes' scale
                % factors)
                lBoundSame = max(0, (probMarg(b,a)-probMargPrev(b,a)) ...
                    / (probNew(b,a)*(1-probMargPrev(b,a))));
                uBoundSame = min(1/probNew(b,a), probMarg(b,a) ...
                    / (probNew(b,a)*(1-probMargPrev(b,a))));

                scaleFactorNew = lBound + (1 - lBoundSame)/(uBoundSame ...
                    - lBoundSame)*(uBound - lBound);
                % Substitute scale factor for new edge into equation of a
                % line to find scale factor for existing edge
                slope = probNew(b,a)*(probMargPrev(bPrev,aPrev) - 1) ...
                    / (probExist(b,a)*probMargPrev(bPrev,aPrev));
                intercept = probMarg(b,a) / (probExist(b,a) ...
                    * probMargPrev(bPrev,aPrev));
                scaleFactorExist = slope*scaleFactorNew + intercept;
                
                scaleFactors(zeroBlockTo,node) = scaleFactorNew;
                scaleFactors(oneBlockTo,node) = scaleFactorExist;
            else
                scaleFactors(zeroBlockFrom',node) = scaleFactorNew;
                scaleFactors(oneBlockFrom',node) = scaleFactorExist;
            end
        end
    end
end

% No self-edges allowed so keep scale factors along the diagonal at 1
scaleFactors(node,node) = 1;

end

