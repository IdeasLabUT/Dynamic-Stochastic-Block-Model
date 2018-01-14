function [classPerm,bestPerm] = permuteClasses(class1,class2)
%permuteClasses Permute classes between two partitions for maximum agreement
%   [classPerm,bestPerm] = permuteClasses(class1,class2)
%
%   Permutes class labels in class membership vector class2 for maximum
%   agreement with labels in class1. This function uses an exhaustive
%   search of all possible permutations and should only be used for a small
%   number of classes.
%
%   Inputs:
%   class1 - Reference class membership vector. Can also be an n-by-tMax
%            matrix where n is the number of objects (nodes) and tMax is
%            the number of time steps.
%   class2 - Class membership vector to permute. Can also be an n-by-tMax
%            matrix.
%
%   Outputs:
%   classPerm - Permuted version of class2. The permutation that results in
%               the largest number of agreements (same class assignments)
%               with class1 is chosen.
%   bestPerm - Best permutation vector. The i-th entry of best perm denotes
%              the class in class2 that was permuted to become class i.

% Author: Kevin S. Xu

[n,tMax] = size(class1);
k = max(class1(:));

allPerms = perms(1:k);
numPerms = size(allPerms,1);
agree = zeros(numPerms,tMax);

for idx = 1:numPerms
	perm = allPerms(idx,:);
	for t = 1:tMax
		class2Curr = cluvec2mat(class2(:,t));
		if ~isequal(size(class2Curr),[n k])	% Correction for empty classes
			class2Curr(n,k) = 0;
		end
		class2CurrPerm = clumat2vec(class2Curr(:,perm));
		agree(idx,t) = sum(class1(:,t)==class2CurrPerm);
	end
end

[~,bestPermIdx] = max(sum(agree,2));
bestPerm = allPerms(bestPermIdx,:);
classPerm = zeros(n,tMax);
for t = 1:tMax
	class2Curr = cluvec2mat(class2(:,t));
	if ~isequal(size(class2Curr),[n k])	% Correction for empty classes
		class2Curr(n,k) = 0;
	end
	classPerm(:,t) = clumat2vec(class2Curr(:,bestPerm));
end
