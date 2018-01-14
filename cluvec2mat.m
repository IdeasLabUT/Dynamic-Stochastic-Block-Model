function Clu_mat = cluvec2mat(clu_vect)
% CLUVEC2MAT	Convert cluster membership vector to matrix.
% 
% cluvec2mat(clu_vect) converts the cluster membership vector
% clu_vect to a binary cluster membership matrix (rows corresponding to 
% samples, columns corresponding to clusters) with entry (i,j) = 1 if
% sample i is in cluster j. clu_vect should be a vector with entries
% indicating the cluster number of each sample.
% 
% Author: Kevin Xu

n = length(clu_vect);
k = max(clu_vect);
Clu_mat = zeros(n,k);

for clu = 1:k
	Clu_mat(clu_vect==clu,clu) = 1;
end
