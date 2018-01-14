function Opt = setDefaultParam(Opt,defaultFields,defaultValues)
%setDefaultParam Set unspecified optional parameters to default values
%   Opt = setDefaultParam(Opt,defaultFields,defaultVals)
%
%   Inputs:
%   Opt - Struct of optional parameters. Set to empty struct to use all
%         default values
%   defaultFields - Names (fields in struct) of optional parameters,
%                   specified as a length n cell array
%   defaultValues - Values of optional parameters, specified as a length n
%                   cell array where the i-th element corresponds to the
%                   value for the i-th field
%
%   Outputs:
%   Opt - Updated struct of optional parameter values with default values
%         applied where necessary

% Author: Kevin S. Xu

nFields = length(defaultFields);
assert(length(defaultValues)==nFields, ...
    'Number of specified values must match number of specified fields');

for i = 1:nFields
    % If field is not part of Opt, create the field and use default value
    if ~isfield(Opt,defaultFields{i})
        Opt.(defaultFields{i}) = defaultValues{i};
    end
end

end

