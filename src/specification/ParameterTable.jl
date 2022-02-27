############################################################################
### Types
############################################################################

mutable struct ParameterTable{SV, BV, FV, SyV}
    latent_vars::SV
    observed_vars::SV
    sorted_vars::SV
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

############################################################################
### Constructors
############################################################################

# constuct an empty table
function ParameterTable(disambig::Nothing)

    from = Vector{String}()
    parameter_type = Vector{String}()
    to = Vector{String}()
    free = Vector{Bool}()
    value_fixed = Vector{Float64}()
    label = Vector{String}()
    start = Vector{Float64}()
    estimate = Vector{Float64}()
    identifier = Vector{Symbol}()

    latent_vars = Vector{String}()
    observed_vars = Vector{String}()
    sorted_vars = Vector{String}()

    return ParameterTable(
        latent_vars,
        observed_vars,
        sorted_vars,
        from,
        parameter_type,
        to,
        free,
        value_fixed,
        label,
        start,
        estimate,
        identifier
    )

end

#= function ParameterTable(;graph::StenoGraph, kwargs...)
    
end =#

############################################################################
### Convert to other types
############################################################################

import Base.Dict

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
### Additional Methods
############################################################################

# Iteration ----------------------------------------------------------------

Base.getindex(partable::ParameterTable, i::Int) =
    (partable.from[i], 
    partable.parameter_type[i], 
    partable.to[i], 
    partable.free[i], 
    partable.value_fixed[i], 
    partable.label[i],
    partable.identifier[i])

Base.length(partable::ParameterTable) = length(partable.from)

# Sorting -------------------------------------------------------------------

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

# add a row -------------------------------------------------------------------

import Base.push!

function push!(partable::ParameterTable, from, parameter_type, to, free, value_fixed, label, start, estimate, identifier)
    
    push!(partable.from, from)
    push!(partable.parameter_type, parameter_type)
    push!(partable.to, to)
    push!(partable.free, free)
    push!(partable.value_fixed, value_fixed)
    push!(partable.label, label)
    push!(partable.start, start)
    push!(partable.estimate, estimate)
    push!(partable.identifier, identifier)

end

############################################################################
### Update Fitted Model
############################################################################

# update estimates ---------------------------------------------------------

function update_estimate!(partable::ParameterTable, sem_fit::SemFit)
    for (i, identifier) in enumerate(partable.identifier)
        if identifier == :const 
        else
            partable.estimate[i] = sem_fit.solution[sem_fit.model.imply.identifier[identifier]] 
        end
    end
    return partable
end


# update starting values -----------------------------------------------------

function update_start!(partable::ParameterTable, sem_fit::SemFit)
    for (i, identifier) in enumerate(partable.identifier)
        if identifier == :const 
        else
            partable.start[i] = sem_fit.model.imply.start_val[sem_fit.model.imply.identifier[identifier]] 
        end
    end
    return partable
end

function update_start!(partable::ParameterTable, model::Sem{O, I, L, D}) where{O, I, L, D}
    for (i, identifier) in enumerate(partable.identifier)
        if identifier == :const 
        else
            partable.start[i] = model.imply.start_val[model.imply.identifier[identifier]]
        end
    end
    return partable
end

# update estimates and starting values ----------------------------------------

function update_partable!(partable::ParameterTable, sem_fit::SemFit)
    update_start!(partable, sem_fit)
    update_estimate!(partable, sem_fit)
    return partable
end