############################################################################################
### Types
############################################################################################

mutable struct EnsembleParameterTable{C} <: AbstractParameterTable
    tables::C
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
function EnsembleParameterTable(::Nothing)
    tables = Dict{Symbol, ParameterTable}()
    return EnsembleParameterTable(tables)
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
### get parameter table from RAMMatrices
############################################################################################

function EnsembleParameterTable(args...; groups)
    partable = EnsembleParameterTable(nothing)

    for (group, ram_matrices) in zip(groups, args)
        push!(partable.tables, group => ParameterTable(ram_matrices))
    end

    return partable
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

# get the vector of all parameters in the table
# the position of the parameter is based on its first appearance in the table (and the ensemble)
function params(partable::EnsembleParameterTable)
    params = mapreduce(vcat, values(partable.tables)) do tbl
        tbl.columns[:param]
    end
    return filter!(!=(:const), unique!(params)) # exclude constants
end

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
