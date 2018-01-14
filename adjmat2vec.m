function adj_vect = adjmat2vec(Adj,directed,self_edges)
% adjmat2vec(Adj) converts an n-by-n adjacency matrix of an undirected graph 
% without self-edges into a length n(n-1)/2 vector.
% 
% adjmat2vec(Adj,directed) converts the adjacency matrix of a directed graph
% into a length n(n-1) vector if directed is set to true.
% 
% adjmat2vec(Adj,directed,self_edges) converts the adjacency matrix of a
% graph (either directed or undirected) with self-edges into an
% appropriately sized vector if self_edges is set to true.
% 
% Adj can also be a collection of adjacency matrices, stored as a 3-D
% matrix where the third dimension indexes the adjacency matrices.
% 
% Author: Kevin Xu

if nargin < 3
	self_edges = false;
end

if nargin < 2
	directed = false;
end

[m,n,t_max] = size(Adj);
if m~= n
	error('Adjacency matrix must be square')
end
if directed == false
	for t = 1:t_max
		if ~isequal(Adj(:,:,t),Adj(:,:,t)')
			error('Adjacency matrix of undirected graph must be symmetric')
		end
	end
end

% As of MATLAB 2011a, there is no support for sparse 3-D matrices so we can
% assume t_max = 1 if Adj is sparse
if issparse(Adj)
	if directed == false
		% Convert lower triangular part of adjacency matrix into vector,
		% beginning on the diagonal if self-edges are included, and one
		% below the diagonal if self-edges are not included.
		if self_edges == false
			offset = 1;
		else
			offset = 0;
		end
		adj_vect = sparse(0,0);
		for col = 1:n-offset
			adj_vect = [adj_vect; Adj(col+offset:end,col)]; %#ok<*AGROW>
		end
	else
		if self_edges == false
			% Create stacked vector consisting of upper triangular part
			% followed by lower triangular part.
			adj_vect = sparse(0,0);
			% Above the diagonal
			for col = 2:n
				adj_vect = [adj_vect; Adj(1:col-1,col)];
			end
			% Below the diagonal
			for col = 1:n-1
				adj_vect = [adj_vect; Adj(col+1:end,col)];
			end
		else
			adj_vect = reshape(Adj,n^2,1);
		end
	end
else
	if directed == false
		if self_edges == false
			p = n*(n-1)/2;
		else
			p = n*(n+1)/2;
		end
	else
		if self_edges == false
			p = n*(n-1);
		else
			p = n^2;
		end
	end
	adj_vect = zeros(p,t_max);
	for t = 1:t_max
		Adj_t = Adj(:,:,t);
		if directed == false
			% Convert lower triangular part of adjacency matrix into vector,
			% beginning on the diagonal if self-edges are included, and one
			% below the diagonal if self-edges are not included.
			if self_edges == false
				offset = -1;
			else
				offset = 0;
			end
			adj_vect(:,t) = Adj_t(tril(true(n),offset));
		else
			if self_edges == false
				% Create stacked vector consisting of upper triangular part
				% followed by lower triangular part.
				adj_vect(:,t) = [Adj_t(triu(true(n),1)); ...
					Adj_t(tril(true(n),-1))];
			else
				adj_vect(:,t) = reshape(Adj_t,p,1);
			end
		end
	end
end
