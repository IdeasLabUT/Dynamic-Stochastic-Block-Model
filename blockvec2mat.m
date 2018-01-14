function blockMat = blockvec2mat(blockVec,directed)
%blockvec2mat Convert block vector to matrix
%   blockMat = blockvec2mat(blockVec,directed)
%
%   blockvec2mat converts a vector representation of a k x k block matrix
%   back into its matrix form, where k denotes the number of classes in a
%   stochastic block model. If the input is a time series of block vectors,
%   each one will be converted into a block matrix.
%
%   Inputs:
%   blockVec - Length p vector of entries to convert to k x k matrix. p
%              must be k*(k+1)/2 for undirected graphs and k^2 for directed
%              graphs. Can also be a p x tMax matrix where each column of
%              blockVec will be converted to a matrix.
%   directed - Whether blockVec comes from a directed graph (default: false)
%
%   Outputs:
%   blockMat - k x k matrix of entries converted from blockVec. If blockVec
%              is a matrix, the output will be a k x k x tMax 3-D array
%              where each slice is converted from each column of blockVec.

% Author: Kevin S. Xu

if nargin < 2
    directed = false;
end

if isvector(blockVec)
    % Ensure that blockVec is a row vector
    blockVec = reshape(blockVec,length(blockVec),1);
end

[p,tMax] = size(blockVec);

% Find the correct dimensions of the block matrix depending on whether
% graph is directed or undirected
if directed == true
    k = sqrt(p);
else
    k = (-1+sqrt(1+8*p))/2;
end

if k ~= floor(k)
    error('Block vector is not of correct dimension for the type of graph')
end

if directed == true
    % Directed case: form columns by grabbing consecutive chunks of length
    % k
    blockMat = reshape(blockVec,[k k tMax]);
else
    % Undirected case: map entries to lower diagonal of matrix, including
    % diagonal, then copy entries to upper diagonal
    lMask = tril(true(k));
    uMask = triu(true(k),1);
    blockMat = zeros([k k tMax]);
    for t = 1:tMax
        blockMatCurr = zeros(k,k);
        blockMatCurr(lMask) = blockVec(:,t);
        blockMatCurrTrans = blockMatCurr';
        blockMatCurr(uMask) = blockMatCurrTrans(uMask);
        blockMat(:,:,t) = blockMatCurr;
    end
end

end

