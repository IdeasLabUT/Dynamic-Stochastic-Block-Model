function [durations,intervals,Opt] = calcEdgeDurationsIntervals(adj,Opt)
%calcEdgeDurationsIntervals Calculate durations and intervals of edges
%   [durations,intervals] = calcEdgeDurationsIntervals(adj,directed)
%
%   calcEdgeDurationsIntervals calculates the durations and intervals of
%   edges in a dynamic network.
%
%   Inputs:
%   adj - 3-D array of adjacency matrices, where each slice denotes the
%         adjacency matrix at a particular time step
%   Opt - Struct of optional parameters. Set to an empty struct to use all
%         default values.
%
%   Optional inputs (specified as fields of Opt [default in brackets]):
%   'directed' - Whether the graph is directed (set to true or false)
%               [ false ]
%   'output' - Level of output to print to command window. Higher values
%              result in more output, and 0 results in no output. Set to 2
%              or higher to print to the command window every 100 nodes.
%              [ 0 ]
%
%   Outputs:
%   durations - Vector of durations of all edges appearing in the dynamic
%               network
%   intervals - Vector of intervals between all edges in the dynamic
%               network

% Author: Kevin S. Xu

% Set defaults for optional parameters if necessary
defaultFields = {'directed','output'};
defaultValues = {false,0};
Opt = setDefaultParam(Opt,defaultFields,defaultValues);
directed = Opt.directed;
output = Opt.output;

[n,~,tMax] = size(adj);
adjDiff = zeros(n,n,tMax-1);
for t = 1:tMax-1
    adjDiff(:,:,t) = adj(:,:,t+1)-adj(:,:,t);
end

maxNumTrans = sum(adjDiff(:)~=0);
durations = zeros(maxNumTrans,1);
intervals = zeros(maxNumTrans,1);
% Counters for duration and interval vectors
durCtr = 1;
intCtr = 1;
for row = 1:n
    if (output > 1) && (mod(row,100) == 0)
        disp(['Calculating edge durations and intervals for node ' ...
            int2str(row)])
    end
    
    if directed == true
        startCol = 1;
    else
        startCol = row+1;
    end
    
    for col = startCol:n
        % Times at which edges are added and removed
        addTimes = find(adjDiff(row,col,:) == 1)'+1;
        removeTimes = find(adjDiff(row,col,:) == -1)'+1;

        if (adj(row,col,1) == 1) && (adj(row,col,tMax) == 1)
            numDur = length(addTimes)+1;
            numInt = length(addTimes);
            durations(durCtr:durCtr+numDur-1) = [removeTimes tMax+1] ...
                - [1 addTimes];
            intervals(intCtr:intCtr+numInt-1) = addTimes - removeTimes;
        elseif (adj(row,col,1) == 1) && (adj(row,col,tMax) == 0)
            numDur = length(removeTimes);
            numInt = length(removeTimes);
            durations(durCtr:durCtr+numDur-1) = removeTimes - [1 addTimes];
            intervals(intCtr:intCtr+numInt-1) = [addTimes tMax+1] ...
                - removeTimes;
        elseif (adj(row,col,1) == 0) && (adj(row,col,tMax) == 1)
            numDur = length(addTimes);
            numInt = length(addTimes);
            durations(durCtr:durCtr+numDur-1) = [removeTimes tMax+1] ...
                - addTimes;
            intervals(intCtr:intCtr+numInt-1) = addTimes - [1 removeTimes];
        else
            numDur = length(addTimes);
            numInt = length(addTimes)+1;
            durations(durCtr:durCtr+numDur-1) = removeTimes - addTimes;
            intervals(intCtr:intCtr+numInt-1) = [addTimes tMax+1] ...
                - [1 removeTimes];
        end
        durCtr = durCtr+numDur;
        intCtr = intCtr+numInt;
    end
end

durations = durations(durations > 0);
% Remove edges that never appeared
intervals = intervals((intervals > 0) & (intervals < tMax));

end

