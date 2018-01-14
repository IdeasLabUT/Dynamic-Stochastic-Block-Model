function Adj_mat = adjvec2mat(adj,directed,self_edges)
% adjvec2mat(adj) converts the vector representation of an undirected graph
% without self-edges into an adjacency matrix.
% 
% adjvec2mat(adj,directed) converts the vector representation of a directed
% graph without self-edges into an adjacency matrix if directed is set to true.
% 
% adjvec2mat(adj,directed,self_edges) converts the vector representation
% of a graph (either directed or undirected) with self-edges into an
% adjacency matrix if self_edges is set to true.
% 
% adj can also be a collection of adjacency vectors, stored as a matrix
% where each column corresponds to an adjacency vector.
% 
% Author: Kevin Xu

if nargin < 3
	self_edges = false;
end

if nargin < 2
	directed = false;
end

[p,t_max] = size(adj);
% If adj is a row vector, convert it to a column vector
if p == 1
	p = t_max;
	t_max = 1;
	adj = adj';
end
% Find the correct dimensions of the adjacency matrix depending on whether
% adj is representing a directed or undirected graph with or without
% self-edges.
if directed == false
	if self_edges == false
		n = (1+sqrt(1+8*p))/2;
	else
		n = (-1+sqrt(1+8*p))/2;
	end
else
	if self_edges == false
		n = (1+sqrt(1+4*p))/2;
	else
		n = sqrt(p);
	end
end

if n~=floor(n)
	error('Adjacency vector is not of correct dimension for the type of graph')
end

Adj_mat = zeros(n,n,t_max);
for t = 1:t_max
	Adj_mat_t = zeros(n,n);
	if directed == false
		% Convert lower triangular part of adjacency matrix into vector,
		% beginning on the diagonal if self-edges are included, and one
		% below the diagonal if self-edges are not included.
		if self_edges == false
			offset = -1;
		else
			offset = 0;
		end
		Adj_mat_t(tril(true(n),offset)) = adj(:,t);
		% Duplicate lower triangular part to upper triangular part by taking
		% transpose then re-assigning lower triangular part
		Adj_mat_t = Adj_mat_t';
		Adj_mat_t(tril(true(n),offset)) = adj(:,t);
	else
		if self_edges == false
			% First portion of adjacency vector corresponds to upper triangular
			% part, and second portion to lower triangular part
			Adj_mat_t(triu(true(n),1)) = adj(1:p/2,t);
			Adj_mat_t(tril(true(n),-1)) = adj(p/2+1:p,t);
		else
			Adj_mat_t = reshape(adj(:,t),n,n);
		end
	end
	Adj_mat(:,:,t) = Adj_mat_t;
end
