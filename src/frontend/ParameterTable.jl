mutable struct ParameterTable
    latent_vars
    observed_vars
    from
    parameter_type
    to
    free
    value_fixed
    label
    start
    estimate
end

Base.getindex(partable::ParameterTable, i::Int) =
    (partable.from[i], 
    partable.parameter_type[i], 
    partable.to[i], 
    partable.free[i], 
    partable.value_fixed[i], 
    partable.label[i])

Base.length(partable::ParameterTable) = length(partable.from)