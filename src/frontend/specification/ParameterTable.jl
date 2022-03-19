############################################################################
### Types
############################################################################

mutable struct ParameterTable{C, l, V}
    columns::C
    len::l # length
    variables::V
    #latent_vars::SV
    #observed_vars::SV
    #sorted_vars::SV
    #from::SV
    #parameter_type::SV
    #to::SV
    #free::BV
    #value_fixed::FV
    #label::SV
    #start::FV
    #estimate::FV
    #identifier::SyV
    #group::SV
    #start_partable::BV
end

############################################################################
### Constructors
############################################################################

# constuct an empty table
function ParameterTable(disambig::Nothing)

    columns = Dict{Symbol, Any}(
        :from => Vector{Symbol}(),
        :parameter_type => Vector{Symbol}(),
        :to => Vector{Symbol}(),
        :free => Vector{Bool}(),
        :value_fixed => Vector{Float64}(),
        :label => Vector{Symbol}(),
        :start => Vector{Float64}(),
        :estimate => Vector{Float64}(),
        :identifier => Vector{Symbol}(),
    )

    variables = Dict{Symbol, Any}(
        :latent_vars => Vector{Symbol}(),
        :observed_vars => Vector{Symbol}(),
        :sorted_vars => Vector{Symbol}()
    )

    return ParameterTable(columns, 0, variables)
end

############################################################################
### Convert to other types
############################################################################

import Base.Dict

function Dict(partable::ParameterTable)
    return partable.columns
end

function DataFrame(
        partable::ParameterTable; 
        columns = [:from, :parameter_type, :to, :free, :value_fixed, :label, :start, :estimate, :identifier])
    out = DataFrame([key => partable.columns[key] for key in columns])
    return DataFrame(out)
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_columns = [
        :from,
        :parameter_type,
        :to,
        :free,
        :value_fixed,
        :label,
        :start,
        :estimate,
        :identifier]
    existing_columns = [haskey(partable.columns, key) for key in relevant_columns]
    
    as_matrix = hcat([partable.columns[key] for key in relevant_columns[existing_columns]]...)
    pretty_table(
        io, 
        as_matrix,
        header = (
            relevant_fields,
            eltype.(getproperty.([partable], relevant_fields))
        ),
        tf = PrettyTables.tf_compact)

    if haskey(partable.variables, :latent_vars)
        print(io, "Latent Variables:    $(partable.variables[:latent_vars]) \n")
    end
    if haskey(partable.variables, :observed_vars)
        print(io, "Observed Variables:  $(partable.variables[:observed_vars]) \n")
    end
end

############################################################################
### Additional Methods
############################################################################

# Iteration ----------------------------------------------------------------

Base.getindex(partable::ParameterTable, i::Int) =
    (partable.columns[:from][i], 
    partable.columns[:parameter_type][i], 
    partable.columns[:to][i], 
    partable.columns[:free][i], 
    partable.columns[:value_fixed][i], 
    partable.columns[:label][i],
    partable.columns[:identifier][i])

Base.length(partable::ParameterTable) = partable.len

# Sorting -------------------------------------------------------------------

import Base.sort!, Base.sort

function sort!(partable::ParameterTable)

    variables = [partable.variables[:latent_vars]; partable.variables[:observed_vars]]

    is_regression = partable.columns[:parameter_type] .== :→

    to = partable.columns[:to][is_regression]
    from = partable.columns[:from][is_regression]

    sorted_variables = Vector{Symbol}()

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

    push!(partable.variables, :sorted_vars => sorted_variables)

end

function sort(partable::ParameterTable)
    new_partable = deepcopy(partable)
    sort!(new_partable)
    return new_partable
end

# add a row -------------------------------------------------------------------

#= import Base.push!

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

end =#

############################################################################
### Update Fitted Model
############################################################################

# update estimates ---------------------------------------------------------

function update_estimate!(partable::ParameterTable, sem_fit::SemFit)
    for (i, identifier) in enumerate(partable.columns[:identifier])
        if !(identifier == :const)
            partable.columns[:estimate][i] = sem_fit.solution[sem_fit.model.imply.identifier[identifier]]
        end
    end
    return partable
end


# update starting values -----------------------------------------------------

function update_start!(partable::ParameterTable, sem_fit::SemFit)
    for (i, identifier) in enumerate(partable.columns[:identifier])
        if !(identifier == :const)
            partable.columns[:start][i] = sem_fit.start_val[sem_fit.model.imply.identifier[identifier]] 
        end
    end
    return partable
end

function update_start!(partable::ParameterTable, model::Sem{O, I, L, D}, start_val) where{O, I, L, D}
    if !(start_val isa Vector)
        start_val = start_val(model)
    end
    for (i, identifier) in enumerate(partable.columns[:identifier])
        if !(identifier == :const)
            partable.columns[:start][i] = start_val[model.imply.identifier[identifier]]
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