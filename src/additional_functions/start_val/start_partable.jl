"""
    start_parameter_table(model; parameter_table)

Return a vector of starting values taken from `parameter_table`.
"""
function start_parameter_table end

# splice model and loss functions
function start_parameter_table(model::AbstractSemSingle; kwargs...)
    return start_parameter_table(
        model.observed,
        model.imply,
        model.optimizer,
        model.loss.functions...;
        kwargs...,
    )
end

# RAM(Symbolic)
function start_parameter_table(observed, imply, optimizer, args...; kwargs...)
    return start_parameter_table(ram_matrices(imply); kwargs...)
end

function start_parameter_table(ram::RAMMatrices; partable::ParameterTable, kwargs...)
    start_val = zeros(0)

    param_indices = Dict(param => i for (i, param) in enumerate(params(ram)))
    start_col = partable.columns[:start]

    for (i, param) in enumerate(partable.columns[:param])
        par_ind = get(param_indices, param, nothing)
        if !isnothing(par_ind)
            par_start = start_col[i]
            isfinite(par_start) && (start_val[i] = par_start)
        else
            throw(ErrorException("Parameter $(param) is not in the parameter table."))
        end
    end

    return start_val
end
