function stateCov = generateStateCov(k,sIn,sOut,directed)
%generateStateCov Generate state covariance matrix for dynamic SBM
%   stateCov = generateStateCov(k,sIn,sOut,directed)
%
%   Generates state covariance matrix for dynamic SBM based on two
%   parameters: sIn, the variance for all states, and sOut, the covariance
%   between any state and any neighboring state. Each state corresponds to
%   an SBM block (a,b), and a neighboring state is any SBM block (a',b')
%   such that either a' = a or b' = b but not both. The covariance between
%   all other states is taken to be 0.
%
%   Inputs:
%   k - Number of classes in the SBM
%   sIn - Variance for all states
%   sOut - Covariance between neighboring states
%
%   Optional inputs [ default in brackets ]:
%   directed - Whether the graph is directed [ false ]
%
%   Outputs:
%   stateCov - State covariance matrix

% Author: Kevin S. Xu

if nargin < 4
    directed = false;
end

if directed == false
	p = k*(k+1)/2;
else
	p = k^2;
end

% Set all variances to s_in
stateCov = sIn*eye(p,p);

% mapMat maps matrix indices to vector indices
mapMat = blockvec2mat(1:p,directed);

for row = 1:k
	if directed == false
        endIdx = row;
	else
		endIdx = k;
	end
	for col = 1:endIdx
		covEntry = mapMat(row,col);
		% Indices of other blocks on same row as current block
		covRow = setdiff(mapMat(row,:),covEntry);
		stateCov(covEntry,covRow) = sOut;
		stateCov(covRow,covEntry) = sOut;
		
		if row ~= col
			% Indices of other blocks on same column as current block
			covCol = setdiff(mapMat(:,col),covEntry);
			stateCov(covEntry,covCol) = sOut;
			stateCov(covCol,covEntry) = sOut;
		end
	end
end

end

