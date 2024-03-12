# specification colnames (only observed)
function get_colnames(specification::ParameterTable)
    colnames = isempty(specification.variables.sorted) ?
        specification.variables.observed :
        filter(in(Set(specification.variables.observed)),
               specification.variables.sorted)
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