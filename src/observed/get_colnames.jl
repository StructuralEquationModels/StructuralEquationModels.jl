# specification colnames (only observed)
function get_colnames(specification::ParameterTable)
    colnames =
        isempty(specification.sorted_vars) ? specification.observed_vars :
        filter(in(Set(specification.observed_vars)), specification.sorted_vars)
    return colnames
end

function get_colnames(specification::RAMMatrices)
    if isnothing(specification.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        colnames = specification.colnames[specification.F_ind]
        return colnames
    end
end

function get_colnames(specification::Nothing)
    return nothing
end
