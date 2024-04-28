############################################################################################
### Types
############################################################################################

mutable struct ParameterTable{C, V} <: AbstractParameterTable
    columns::C
    variables::V
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
function ParameterTable(::Nothing)
    columns = Dict{Symbol, Any}(
        :from => Vector{Symbol}(),
        :parameter_type => Vector{Symbol}(),
        :to => Vector{Symbol}(),
        :free => Vector{Bool}(),
        :value_fixed => Vector{Float64}(),
        :start => Vector{Float64}(),
        :estimate => Vector{Float64}(),
        :identifier => Vector{Symbol}(),
        :start => Vector{Float64}(),
    )

    variables = Dict{Symbol, Any}(
        :latent_vars => Vector{Symbol}(),
        :observed_vars => Vector{Symbol}(),
        :sorted_vars => Vector{Symbol}(),
    )

    return ParameterTable(columns, variables)
end

############################################################################################
### Convert to other types
############################################################################################

import Base.Dict

function Dict(partable::ParameterTable)
    return partable.columns
end

function DataFrame(partable::ParameterTable; columns = nothing)
    if isnothing(columns)
        columns = keys(partable.columns)
    end
    out = DataFrame([key => partable.columns[key] for key in columns])
    return DataFrame(out)
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_columns = [
        :from,
        :parameter_type,
        :to,
        :free,
        :value_fixed,
        :start,
        :estimate,
        :se,
        :identifier,
    ]
    existing_columns = [haskey(partable.columns, key) for key in relevant_columns]

    as_matrix =
        hcat([partable.columns[key] for key in relevant_columns[existing_columns]]...)
    pretty_table(
        io,
        as_matrix,
        header = (
            relevant_columns[existing_columns],
            eltype.([partable.columns[key] for key in relevant_columns[existing_columns]]),
        ),
        tf = PrettyTables.tf_compact,
    )

    if haskey(partable.variables, :latent_vars)
        print(io, "Latent Variables:    $(partable.variables[:latent_vars]) \n")
    end
    if haskey(partable.variables, :observed_vars)
        print(io, "Observed Variables:  $(partable.variables[:observed_vars]) \n")
    end
end

############################################################################################
### Additional Methods
############################################################################################

# Iteration --------------------------------------------------------------------------------

Base.getindex(partable::ParameterTable, i::Int) = (
    partable.columns[:from][i],
    partable.columns[:parameter_type][i],
    partable.columns[:to][i],
    partable.columns[:free][i],
    partable.columns[:value_fixed][i],
    partable.columns[:identifier][i],
)

function Base.length(partable::ParameterTable)
    len = missing
    for key in keys(partable.columns)
        len = length(partable.columns[key])
        break
    end
    return len
end

# Sorting ----------------------------------------------------------------------------------

struct CyclicModelError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, e::CyclicModelError) = print(io, e.msg)

import Base.sort!, Base.sort

function sort!(partable::ParameterTable)
    variables = [partable.variables[:latent_vars]; partable.variables[:observed_vars]]

    is_regression =
        (partable.columns[:parameter_type] .== :→) .&
        (partable.columns[:from] .!= Symbol("1"))

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

        if !acyclic
            throw(CyclicModelError("your model is cyclic and therefore can not be ordered"))
        end
        acyclic = false

        if length(variables) == 0
            sorted = true
        end
    end

    push!(partable.variables, :sorted_vars => sorted_variables)

    return partable
end

function sort(partable::ParameterTable)
    new_partable = deepcopy(partable)
    sort!(new_partable)
    return new_partable
end

# add a row --------------------------------------------------------------------------------

import Base.push!

function push!(partable::ParameterTable, d::AbstractDict)
    for key in keys(d)
        push!(partable.columns[key], d[key])
    end
end

push!(partable::ParameterTable, d::Nothing) = nothing

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------

function update_partable!(
    partable::ParameterTable,
    model_identifier::AbstractDict,
    vec,
    column,
)
    new_col = Vector{eltype(vec)}(undef, length(partable))
    for (i, identifier) in enumerate(partable.columns[:identifier])
        if !(identifier == :const)
            new_col[i] = vec[model_identifier[identifier]]
        elseif identifier == :const
            new_col[i] = zero(eltype(vec))
        end
    end
    push!(partable.columns, column => new_col)
    return partable
end

"""
    update_partable!(partable::AbstractParameterTable, sem_fit::SemFit, vec, column)
    
Write `vec` to `column` of `partable`.

# Arguments
- `vec::Vector`: has to be in the same order as the `model` parameters
"""
update_partable!(partable::AbstractParameterTable, sem_fit::SemFit, vec, column) =
    update_partable!(partable, identifier(sem_fit), vec, column)

# update estimates -------------------------------------------------------------------------
"""
    update_estimate!(
        partable::AbstractParameterTable,
        sem_fit::SemFit)

Write parameter estimates from `sem_fit` to the `:estimate` column of `partable`
"""
update_estimate!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, sem_fit, sem_fit.solution, :estimate)

# update starting values -------------------------------------------------------------------
"""
    update_start!(partable::AbstractParameterTable, sem_fit::SemFit)
    update_start!(partable::AbstractParameterTable, model::AbstractSem, start_val; kwargs...)

Write starting values from `sem_fit` or `start_val` to the `:estimate` column of `partable`.

# Arguments
- `start_val`: either a vector of starting values or a function to compute starting values
    from `model`
- `kwargs...`: are passed to `start_val`
"""
update_start!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, sem_fit, sem_fit.start_val, :start)

function update_start!(
    partable::AbstractParameterTable,
    model::AbstractSem,
    start_val;
    kwargs...,
)
    if !(start_val isa Vector)
        start_val = start_val(model; kwargs...)
    end
    return update_partable!(partable, identifier(model), start_val, :start)
end

# update partable standard errors ----------------------------------------------------------
"""
    update_se_hessian!(
        partable::AbstractParameterTable,
        sem_fit::SemFit;
        hessian = :finitediff)

Write hessian standard errors computed for `sem_fit` to the `:se` column of `partable`

# Arguments
- `hessian::Symbol`: how to compute the hessian, see [se_hessian](@ref) for more information.

# Examples

"""
function update_se_hessian!(
    partable::AbstractParameterTable,
    sem_fit::SemFit;
    hessian = :finitediff,
)
    se = se_hessian(sem_fit; hessian = hessian)
    return update_partable!(partable, sem_fit, se, :se)
end
