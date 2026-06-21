############################################################################################
### Types
############################################################################################

struct EnsembleParameterTable <: AbstractParameterTable
    tables::Dict{Symbol, ParameterTable}
    param_labels::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
EnsembleParameterTable(::Nothing; param_labels::Union{Nothing, Vector{Symbol}} = nothing) =
    EnsembleParameterTable(
        Dict{Symbol, ParameterTable}(),
        isnothing(param_labels) ? Symbol[] : copy(param_labels),
    )

# convert pairs to dict
EnsembleParameterTable(ps::Pair{K, V}...; param_labels = nothing) where {K, V} =
    EnsembleParameterTable(Dict(ps...); param_labels = param_labels)

# dictionary of SEM specifications
function EnsembleParameterTable(
    spec_ensemble::AbstractDict{K, V};
    param_labels::Union{Nothing, Vector{Symbol}} = nothing,
) where {K, V <: SemSpecification}
    param_labels = if isnothing(param_labels)
        # collect all SEM parameters in ensemble if not specified
        # and apply the set to all partables
        mapreduce(SEM.param_labels, vcat, values(spec_ensemble), init = Symbol[]) |> unique
    else
        copy(param_labels)
    end

    # convert each model specification to ParameterTable
    partables = Dict{Symbol, ParameterTable}(
        Symbol(group) => convert(ParameterTable, spec; param_labels) for
        (group, spec) in pairs(spec_ensemble)
    )
    return EnsembleParameterTable(partables, param_labels)
end

############################################################################################
### Convert to other types
############################################################################################

function Base.convert(::Type{Dict}, partable::EnsembleParameterTable)
    return convert(Dict, partable.tables)
end

function Base.convert(
    ::Type{Dict{K, RAMMatrices}},
    partables::EnsembleParameterTable;
    param_labels::Union{AbstractVector{Symbol}, Nothing} = nothing,
) where {K}
    isnothing(param_labels) || (param_labels = SEM.param_labels(partables))

    return Dict{K, RAMMatrices}(
        K(key) => RAMMatrices(partable; param_labels = param_labels) for
        (key, partable) in pairs(partables.tables)
    )
end

function DataFrames.DataFrame(
    partables::EnsembleParameterTable;
    columns::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    mapreduce(vcat, pairs(partables.tables)) do (key, partable)
        df = DataFrame(partable; columns = columns)
        df[!, :group] .= key
        return df
    end
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::EnsembleParameterTable)
    print(io, "EnsembleParameterTable with groups: ")
    for key in keys(partable.tables)
        print(io, "|", key, "|")
    end
    print(io, "\n")
    for key in keys(partable.tables)
        print("\n")
        print(io, key, ": \n")
        print(io, partable.tables[key])
    end
end

############################################################################################
### Additional Methods
############################################################################################

# Variables Sorting ------------------------------------------------------------------------

function sort_vars!(partables::EnsembleParameterTable)
    for partable in values(partables.tables)
        sort_vars!(partable)
    end

    return partables
end

sort_vars(partables::EnsembleParameterTable) = sort_vars!(deepcopy(partables))

# add a row --------------------------------------------------------------------------------

# do we really need this?
function Base.push!(partable::EnsembleParameterTable, d::AbstractDict, group)
    push!(partable.tables[group], d)
end

Base.getindex(partable::EnsembleParameterTable, group) = partable.tables[group]

############################################################################################
### Update Partable from Fitted Model
############################################################################################

function update_partable!(
    partables::EnsembleParameterTable,
    column::Symbol,
    params::AbstractDict{Symbol},
    default::Any = nothing,
)
    for partable in values(partables.tables)
        update_partable!(partable, column, params, default)
    end
    return partables
end

function update_partable!(
    partables::EnsembleParameterTable,
    column::Symbol,
    param_labels::AbstractVector{Symbol},
    values::AbstractVector,
    default::Any = nothing,
)
    return update_partable!(partables, column, Dict(zip(param_labels, values)), default)
end

############################################################################################
### Additional methods
############################################################################################

function Base.:(==)(p1::EnsembleParameterTable, p2::EnsembleParameterTable)
    out = (p1.tables == p2.tables) && (p1.param_labels == p2.param_labels)
    return out
end
