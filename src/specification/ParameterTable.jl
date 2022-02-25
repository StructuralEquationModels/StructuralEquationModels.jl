Base.@kwdef mutable struct ParameterTable{SV, BV, FV, SyV} <: SemSpecification
    latent_vars::SV
    observed_vars::SV
    sorted_vars::SV = Vector{String}()
    from::SV
    parameter_type::SV
    to::SV
    free::BV
    value_fixed::FV
    label::SV
    start::FV
    estimate::FV
    identifier::SyV
end

Base.getindex(partable::ParameterTable, i::Int) =
    (partable.from[i], 
    partable.parameter_type[i], 
    partable.to[i], 
    partable.free[i], 
    partable.value_fixed[i], 
    partable.label[i])

Base.length(partable::ParameterTable) = length(partable.from)

import Base.Dict

############################################################################
### Constructors
############################################################################

function ParameterTable(;ram_matrices::RAMMatrices, kwargs...)

end

function ParameterTable(;graph::StenoGraph, kwargs...)
    
end


############################################################################
### Convert to other types
############################################################################

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
        :estimate => partable.estimate,
        :identifier => partable.identifier])
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
        :estimate,
        :identifier]
    as_matrix = hcat(getproperty.([partable], relevant_fields)...)
    pretty_table(
        io, 
        as_matrix,
        header = (
            relevant_fields,
            eltype.(getproperty.([partable], relevant_fields))
        ),
        tf = PrettyTables.tf_compact)
    print(io, "Latent Variables:    $(partable.latent_vars) \n")
    print(io, "Observed Variables:  $(partable.observed_vars) \n")
end

############################################################################
### Sorting
############################################################################

import Base.sort!, Base.sort

function sort!(partable::ParameterTable)

    variables = [partable.latent_vars; partable.observed_vars]

    is_regression = partable.parameter_type .== "→"

    to = partable.to[is_regression]
    from = partable.from[is_regression]

    sorted_variables = Vector{String}()

    sorted = false
    while !sorted
        
        acyclic = false
        
        for (i, variable) in enumerate(variables)
            if !(variable ∈ to)
                push!(sorted_variables, variable)
                deleteat!(variables, i)
                delete_edges = from .!= variable
                to = to[delete_edges]
                from = from[delete_edges]
                acyclic = true
            end
        end
        
        if !acyclic error("Your model is cyclic and therefore can not be ordered") end
        acyclic = false

        if length(variables) == 0 sorted = true end
    end

    partable.sorted_vars = sorted_variables

end

function sort(partable::ParameterTable)
    new_partable = deepcopy(partable)
    sort!(new_partable)
    return new_partable
end