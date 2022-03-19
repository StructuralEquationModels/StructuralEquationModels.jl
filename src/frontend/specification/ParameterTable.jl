############################################################################
### Types
############################################################################

mutable struct ParameterTable{C, V}
    columns::C
    variables::V
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
        :start => Vector{Float64}(),
    )

    variables = Dict{Symbol, Any}(
        :latent_vars => Vector{Symbol}(),
        :observed_vars => Vector{Symbol}(),
        :sorted_vars => Vector{Symbol}()
    )

    return ParameterTable(columns, variables)
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
        :se,
        :identifier]
    existing_columns = [haskey(partable.columns, key) for key in relevant_columns]
    
    as_matrix = hcat([partable.columns[key] for key in relevant_columns[existing_columns]]...)
    pretty_table(
        io, 
        as_matrix,
        header = (
            relevant_columns[existing_columns],
            eltype.([partable.columns[key] for key in relevant_columns[existing_columns]])
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

function Base.length(partable::ParameterTable)
    len = missing
    for key in keys(partable.columns)
        len = length(partable.columns[key])
        break
    end
    return len
end

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

    return partable
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
### Update Partable from Fitted Model
############################################################################

# update generic ---------------------------------------------------------------

function update_partable!(partable::ParameterTable, model_identifier::AbstractDict, vec, column)
    if !haskey(partable, column)
        @info "Your parameter table does not have the column $column, so it was added."
        new_col = Vector{eltype(vec)}(undef, length(partable))
        for (i, identifier) in enumerate(partable.columns[:identifier])
            if !(identifier == :const)
                new_col[i] = vec[model_identifier[identifier]]
            elseif identifier == :const
                new_col[i] == zero(eltype(vec))
            end
        end
        push!(partable.columns, column => new_col)
    else
        for (i, identifier) in enumerate(partable.columns[:identifier])
            if !(identifier == :const)
                partable.columns[column][i] = vec[model_identifier[identifier]]
            end
        end
    end
    return partable
end

# update estimates ---------------------------------------------------------

update_estimate!(partable::ParameterTable, sem_fit::SemFit) =
    update_partable!(partable, identifier(sem_fit), sem_fit.solution, :estimate)

# update starting values -----------------------------------------------------

update_start!(partable::ParameterTable, sem_fit::SemFit) =
    update_partable!(partable, identifier(sem_fit), sem_fit.start_val, :start)

function update_start!(partable::ParameterTable, model::AbstractSem, start_val) where{O, I, L, D}
    if !(start_val isa Vector)
        start_val = start_val(model)
    end
    return update_partable!(partable, identifier(model), start_val, :start)
end

# update partable standard errors ---------------------------------------------

function update_se_hessian!(partable::ParameterTable, sem_fit::SemFit; hessian = :finitediff)
    se = se_hessian(sem_fit; hessian = hessian)
    return update_partable!(partable, identifier(model), se, :se)
end