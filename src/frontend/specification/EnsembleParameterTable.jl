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

#= function DataFrame(
        partable::ParameterTable;
        columns = nothing)
    if isnothing(columns) columns = keys(partable.columns) end
    out = DataFrame([key => partable.columns[key] for key in columns])
    return DataFrame(out)
end =#

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
import Base.push!

function push!(partable::EnsembleParameterTable, d::AbstractDict, group)
    push!(partable.tables[group], d)
end

push!(partable::EnsembleParameterTable, d::Nothing, group) = nothing

Base.getindex(partable::EnsembleParameterTable, group) = partable.tables[group]

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------
function update_partable!(
    partable::EnsembleParameterTable,
    param_indices::AbstractDict,
    vec,
    column,
)
    for k in keys(partable.tables)
        update_partable!(partable.tables[k], param_indices, vec, column)
    end
    return partable
end
