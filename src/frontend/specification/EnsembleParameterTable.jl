############################################################################################
### Types
############################################################################################

struct EnsembleParameterTable <: AbstractParameterTable
    tables::Dict{Symbol, ParameterTable}
    params::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
EnsembleParameterTable(::Nothing; params::Union{Nothing, Vector{Symbol}} = nothing) =
    EnsembleParameterTable(
        Dict{Symbol, ParameterTable}(),
        isnothing(params) ? Symbol[] : copy(params),
    )

# dictionary of SEM specifications
function EnsembleParameterTable(
    spec_ensemble::AbstractDict{K, V};
    params::Union{Nothing, Vector{Symbol}} = nothing,
) where {K, V <: SemSpecification}
    params = if isnothing(params)
        # collect all SEM parameters in ensemble if not specified
        # and apply the set to all partables
        unique(mapreduce(SEM.params, vcat, values(spec_ensemble), init = Vector{Symbol}()))
    else
        copy(params)
    end

    # convert each model specification to ParameterTable
    partables = Dict{Symbol, ParameterTable}(
        Symbol(group) => convert(ParameterTable, spec; params) for
        (group, spec) in pairs(spec_ensemble)
    )
    return EnsembleParameterTable(partables, params)
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
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
) where {K}
    isnothing(params) || (params = SEM.params(partables))

    return Dict{K, RAMMatrices}(
        K(key) => RAMMatrices(partable; params = params) for
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

# update generic ---------------------------------------------------------------------------
function update_partable!(
    partable::EnsembleParameterTable,
    params::AbstractVector{Symbol},
    values::AbstractVector,
    column,
)
    for k in keys(partable.tables)
        update_partable!(partable.tables[k], params, values, column)
    end
    return partable
end
