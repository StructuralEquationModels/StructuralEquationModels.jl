############################################################################################
### Types
############################################################################################

struct ParameterTable{C} <: AbstractParameterTable
    columns::C
    observed_vars::Vector{Symbol}
    latent_vars::Vector{Symbol}
    sorted_vars::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

# construct a dictionary with the default partable columns
# optionally pre-allocate data for nrows
empty_partable_columns(nrows::Integer = 0) = Dict{Symbol, Vector}(
    :from => fill(Symbol(), nrows),
    :parameter_type => fill(Symbol(), nrows),
    :to => fill(Symbol(), nrows),
    :free => fill(true, nrows),
    :value_fixed => fill(NaN, nrows),
    :start => fill(NaN, nrows),
    :estimate => fill(NaN, nrows),
    :param => fill(Symbol(), nrows),
)

# construct using the provided columns data or create and empty table
function ParameterTable(
    columns::Dict{Symbol, Vector} = empty_partable_columns();
    observed_vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
    latent_vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    return ParameterTable(columns,
        !isnothing(observed_vars) ? copy(observed_vars) : Vector{Symbol}(),
        !isnothing(latent_vars) ? copy(latent_vars) : Vector{Symbol}(),
        Vector{Symbol}())
end

############################################################################################
### Convert to other types
############################################################################################

function Base.convert(::Type{Dict}, partable::ParameterTable)
    return partable.columns
end

function DataFrames.DataFrame(
    partable::ParameterTable;
    columns::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    if isnothing(columns)
        columns = keys(partable.columns)
    end
    return DataFrame([col => partable.columns[col] for col in columns])
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_columns =
        [:from, :parameter_type, :to, :free, :value_fixed, :start, :estimate, :se, :param]
    shown_columns = filter!(
        col -> haskey(partable.columns, col) && length(partable.columns[col]) > 0,
        relevant_columns,
    )

    as_matrix = mapreduce(col -> partable.columns[col], hcat, shown_columns)
    pretty_table(
        io,
        as_matrix,
        header = (shown_columns, [eltype(partable.columns[col]) for col in shown_columns]),
        tf = PrettyTables.tf_compact,
    )

    print(io, "Latent Variables:    $(partable.latent_vars) \n")
    print(io, "Observed Variables:  $(partable.observed_vars) \n")
end

############################################################################################
### Additional Methods
############################################################################################

# Iteration --------------------------------------------------------------------------------
ParameterTableRow = @NamedTuple begin
    from::Symbol
    parameter_type::Symbol
    to::Symbol
    free::Bool
    value_fixed::Any
    param::Symbol
end

Base.getindex(partable::ParameterTable, i::Integer) = (
    partable.columns[:from][i],
    partable.columns[:parameter_type][i],
    partable.columns[:to][i],
    partable.columns[:free][i],
    partable.columns[:value_fixed][i],
    partable.columns[:param][i],
)

Base.length(partable::ParameterTable) = length(partable.columns[:param])
Base.eachindex(partable::ParameterTable) = Base.OneTo(length(partable))

Base.eltype(::Type{<:ParameterTable}) = ParameterTableRow
Base.iterate(partable::ParameterTable, i::Integer = 1) =
    i > length(partable) ? nothing : (partable[i], i + 1)


# get the vector of all parameters in the table
# the position of the parameter is based on its first appearance in the table (and the ensemble)
params(partable::ParameterTable) =
    filter!(!=(:const), unique(partable.columns[:param]))


# Sorting ----------------------------------------------------------------------------------

struct CyclicModelError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, e::CyclicModelError) = print(io, e.msg)

"""
    sort_vars!(partable::ParameterTable)
    sort_vars!(partables::EnsembleParameterTable)

Sort variables in `partable` so that all independent variables are
before the dependent variables and store it in `partable.sorted_vars`.

If the relations between the variables are acyclic, sorting will
make the resulting `A` matrix in the *RAM* model lower triangular
and allow faster calculations.
"""
function sort_vars!(partable::ParameterTable)
    vars = [
        partable.latent_vars
        partable.observed_vars
    ]

    is_regression = [
        (partype == :→) && (from != Symbol("1")) for (partype, from) in
        zip(partable.columns[:parameter_type], partable.columns[:from])
    ]

    to = partable.columns[:to][is_regression]
    from = partable.columns[:from][is_regression]

    sorted_vars = Vector{Symbol}()

    while !isempty(vars)
        acyclic = false

        for (i, var) in enumerate(vars)
            if !(var ∈ to)
                push!(sorted_vars, var)
                deleteat!(vars, i)
                delete_edges = from .!= var
                to = to[delete_edges]
                from = from[delete_edges]
                acyclic = true
            end
        end

        acyclic ||
            throw(CyclicModelError("your model is cyclic and therefore can not be ordered"))
    end

    copyto!(resize!(partable.sorted_vars, length(sorted_vars)), sorted_vars)

    return partable
end

"""
    sort_vars(partable::ParameterTable)
    sort_vars(partables::EnsembleParameterTable)

Sort variables in `partable` so that all independent variables are
before the dependent variables, and return a copy of `partable`
where the sorted variables are in `partable.sorted_vars`.

See [sort_vars!](@ref) for in-place version.
"""
sort_vars(partable::ParameterTable) = sort_vars!(deepcopy(partable))

# add a row --------------------------------------------------------------------------------

function Base.push!(partable::ParameterTable, d::Union{AbstractDict{Symbol}, NamedTuple})
    for (key, val) in pairs(d)
        push!(partable.columns[key], val)
    end
end

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------

"""
    update_partable!(partable::AbstractParameterTable, params::Vector{Symbol}, values, column)

Write parameter `values` into `column` of `partable`.

The `params` and `values` vectors define the pairs of value
parameters, which are being matched to the `:param` column
of the `partable`.
"""
function update_partable!(
    partable::ParameterTable,
    params::AbstractVector{Symbol},
    values::AbstractVector,
    column::Symbol,
)
    length(params) == length(values) || throw(
        ArgumentError(
            "The length of `params` ($(length(params))) and their `values` ($(length(values))) must be the same",
        ),
    )
    coldata = get!(() -> Vector{eltype(values)}(), partable.columns, column)
    resize!(coldata, length(partable))
    params_index = Dict(zip(params, eachindex(params)))
    for (i, param) in enumerate(partable.columns[:param])
        coldata[i] = param != :const ? values[params_index[param]] : zero(eltype(values))
    end
    return partable
end

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
    return update_partable!(partable, params(model), start_val, :start)
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
