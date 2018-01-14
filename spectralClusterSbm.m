function [c,Z,S,Opt] = spectralClusterSbm(W,k,Opt)
%spectralClusterSbm Estimate SBM class memberships using spectral clustering
%   [c,Z,S,Opt] = spectralClusterSbm(W,k,Opt) returns an estimate of the
%   class memberships of a stochastic block model (SBM) using spectral
%   clustering.
%
%   Inputs:
%   W - Graph adjacency matrix (binary with no self-edges; can be directed,
%       i.e. w(i,j) = 1 denotes edge from i to j, and w(i,j) = 0 denotes
%       absence of edge from i to j)
%   k - Number of classes
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Set to true for directed graph [ false ]
%   'svdType' - Type of singular value decomposition to use, either 'full'
%               or 'sparse' [ 'full' ]
%   'nKmeansReps' - Number of replicates of k-means clustering to run on the
%             spectral embeddings [ 1 ]
%   'embedDim' - Dimensionality of the spectral embedding [ k ]
%
%   Outputs:
%   c - Estimated class membership vector
%   Z - Matrix of node spectral embeddings
%   S - Diagonal matrix of top embedDim eigenvalues
%   Opt - Updated struct of optional parameter values

% Author: Kevin S. Xu
% 
% References:
% Sussman, D. L., Tang, M., Fishkind, D. E., & Priebe, C. E. (2012). 
%   A consistent adjacency spectral embedding for stochastic blockmodel
%   graphs. Journal of the American Statistical Association, 107(499),
%   1119–1128. doi:10.1080/01621459.2012.699795

% Set defaults for optional parameters if necessary
defaultFields = {'directed','svdType','nKmeansReps','embedDim'};
defaultValues = {false,'full',1,k};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
svdType = Opt.svdType;
nKmeansReps = Opt.nKmeansReps;
embedDim = Opt.embedDim;

if strcmp(svdType,'full')
    [U,S,V] = svd(W);
    U = U(:,1:embedDim);
    S = S(1:embedDim,1:embedDim);
    V = V(:,1:embedDim);
elseif strcmp(svdType,'sparse')
    [U,S,V] = svds(W,embedDim);
else
    error('svdType must either be ''full'' or ''sparse''')
end

sqrtS = sqrt(S);
% Scaled singular vectors
Z = U*sqrtS;
% For directed graphs, include also scaled right singular vectors
if directed == true
    Z = [Z V*sqrtS];
end

c = kmeans(Z,k,'EmptyAction','drop','Replicates',nKmeansReps);

end

