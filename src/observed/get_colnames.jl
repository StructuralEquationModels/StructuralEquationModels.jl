# specification colnames
function get_colnames(specification::ParameterTable)
    if !haskey(specification.variables, :sorted_vars) || 
            (length(specification.variables[:sorted_vars]) == 0)
        colnames = specification.variables[:observed_vars]
    else
        is_obs = [var ∈ specification.variables[:observed_vars] for var in specification.variables[:sorted_vars]]
        colnames = specification.variables[:sorted_vars][is_obs]
    end
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