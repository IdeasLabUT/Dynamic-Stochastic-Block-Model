function clu_vect = clumat2vec(clu_mat)
% CLUMAT2VEC	Convert cluster membership matrix to vector
% 
% clumat2vec(clu_mat) converts the cluster membership matrix
% clu_mat (rows corresponding to samples, columns corresponding to
% clusters) to a cluster membership row vector with entries corresponding
% to cluster number. clu_mat should be a binary matrix with entry (i,j) = 1
% if sample i is in cluster j.
% 
% Author: Kevin Xu

[n,k] = size(clu_mat);
clu_vect = zeros(n,1);

for clu = 1:k
	clu_vect(logical(clu_mat(:,clu))) = clu;
end