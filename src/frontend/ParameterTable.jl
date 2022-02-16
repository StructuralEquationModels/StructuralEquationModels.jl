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

function Dict(partable::ParameterTable)
    fields = fieldnames(typeof(partable))
    out = Dict(fields .=> getproperty.([partable], fields))
    return out
end

function DataFrame(partable::ParameterTable)
    out = DataFrame([
        :from => partable.from,
        :parameter_type => partable.parameter_type,
        :to => partable.to,
        :free => partable.free,
        :value_fixed => partable.value_fixed,
        :label => partable.label,
        :start => partable.start,
        :estimate => partable.estimate])
    return out
end

#= function DataFrame(partable::ParameterTable)
    out = DataFrame(Dict(partable))
    return out
end =#

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_fields = [
        :from,
        :parameter_type,
        :to,
        :free,
        :value_fixed,
        :label,
        :start,
        :estimate]
    as_matrix = hcat(getproperty.([partable], relevant_fields)...)
    pretty_table(
        io, 
        as_matrix,
        header = (
            relevant_fields,
            eltype.(getproperty.([partable], relevant_fields))
        ),
        tf = tf_compact)
    print(io, "Latent Variables:    $(partable.latent_vars) \n")
    print(io, "Observed Variables:  $(partable.observed_vars) \n")
end