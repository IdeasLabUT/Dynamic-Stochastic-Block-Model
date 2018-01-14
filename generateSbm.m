function W = generateSbm(c,P,directed)
%generateSbm Generate realization of stochastic block model
%   generateSbm(c,P) generates a graph adjacency matrix using a stochastic
%   block model where the class membership of each node is given by the
%   vector c, and the matrix P contains the probability of forming edges
%   between each pair of classes.
%
%   generateSbm(c,P,directed) allows for the creation of directed graphs by
%   setting directed to true. By default, undirected graphs are created.

% Author: Kevin S. Xu

% Set as undirected graph by default
if nargin < 3, directed = false; end

n = length(c);
k = max(c);

if (size(c,1) ~= 1) && (size(c,2) ~= 1)
    error('c must be a vector')
end
c = reshape(c,n,1);

assert(size(P,1)>=k,'P must be a k-by-k matrix, where k >= max(c)')
if directed == false
    assert(isequal(P,P'),'P must be a symmetric matrix')
end

W = zeros(n,n); % Graph adjacency matrix
for c1 = 1:k
    % First consider connections between two nodes in the same class
    inC1 = (c==c1);
    numC1 = sum(inC1);
    
    % Form edges between nodes in same class only if more than one node in
    % the class
    if numC1 > 1
        if directed == true
            W_Block = zeros(numC1,numC1);
            blockMask = ~diag(true(numC1,1));
            W_Block(blockMask) = bernrnd(P(c1,c1),numC1*(numC1-1),1);
            W(inC1,inC1) = W_Block;
        else
            W(inC1,inC1) = squareform(bernrnd(P(c1,c1), ...
                numC1*(numC1-1)/2,1));
        end
    end
    
    % Form edges between nodes in different classes. Loop start index
    % depends on whether graph is directed or undirected.
    if directed == true
        start = 1;
    else
        start = c1+1;
    end
    for c2 = start:k
        % Diagonal blocks are already considered, so ignore them in the
        % loop
        if c2 == c1
            continue
        end
        
        inC2 = (c==c2);
        numC2 = sum(inC2);
        if numC2 == 0
            continue
        end
        W(inC1,inC2) = bernrnd(P(c1,c2),numC1,numC2);
        if directed == false
            W(inC2,inC1) = W(inC1,inC2)';
        end
    end
end

end

