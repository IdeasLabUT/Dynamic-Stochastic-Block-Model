function blockVec = blockmat2vec(blockMat,directed)
%blockmat2vec Convert block matrix to vector
%   blockVec = blockmat2vec(blockMat,directed)
%
%   blockmat2vec converts a k x k block matrix to an appropriately sized
%   vector representation, where k denotes the number of classes in a
%   stochastic block model. If the input is a time series of block
%   matrices, each one will be converted to a column vector.
%
%   Inputs:
%   blockMat - k x k matrix of entries to convert to vector. Can also be a
%              k x k x tMax array where each slice of blockMat will be
%              converted to a vector.
%   directed - Whether blockMat comes from a directed graph (default: false)
%
%   Output:
%   blockVec - Length p vector of entries converted from blockMat, where
%              p = k*(k+1)/2 for undirected graphs and p = k^2 for directed
%              graphs. If blockMat is a 3-D array, the output will be a
%              matrix where each column is converted from each slice of
%              blockMat.

% Author: Kevin S. Xu

if nargin < 2
    directed = false;
end

[m,n,tMax] = size(blockMat);
assert(m==n, 'Number of rows and columns of block matrices must be identical')
if directed == true
    % Directed case: stack columns of matrix on top of each other
    blockVec = reshape(blockMat,n^2,tMax);
else
    % Undirected case: retain only lower diagonal, including diagonal
    p = n*(n+1)/2;
    % Binary mask of entries to retain from each slice
    mask = repmat(tril(true(n)),[1 1 tMax]);
    blockVec = reshape(blockMat(mask),p,tMax);
end

end

 