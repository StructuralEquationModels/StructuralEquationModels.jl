"""
    start_parameter_table(model; parameter_table)

Return a vector of starting values taken from `parameter_table`.
"""
start_parameter_table(model::AbstractSemSingle;
                      partable::ParameterTable, kwargs...) =
    start_parameter_table(partable)

function start_parameter_table(partable::ParameterTable)
    start_vals = zeros(eltype(partable.columns.start), nparams(partable))
    param_indices = Dict(param => i for (i, param) in enumerate(params(partable)))

    for (param, startval) in zip(partable.columns.param,
                                 partable.columns.start)
        (param == :const) && continue
        par_ind = get(param_indices, param, nothing)
        if !isnothing(par_ind)
            isfinite(startval) && (start_vals[par_ind] = startval)
        else
            throw(ErrorException("Parameter $(param) not found in the model."))
        end
    end

    return start_vals
end