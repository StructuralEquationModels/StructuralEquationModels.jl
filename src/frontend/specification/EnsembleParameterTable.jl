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

import Base.Dict

function Dict(partable::EnsembleParameterTable)
    return partable.tables
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
    for key in keys(partable.tables) print(io, "|", key, "|") end
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

# Sorting ----------------------------------------------------------------------------------

# todo

# add a row --------------------------------------------------------------------------------

# do we really need this?
import Base.push!

function push!(partable::EnsembleParameterTable, d::AbstractDict, group)
    push!(partable.tables[group], d)
end

push!(partable::EnsembleParameterTable, d::Nothing, group) = nothing

# get group --------------------------------------------------------------------------------

get_group(partable::EnsembleParameterTable, group) = get_group(partable.tables, group)

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------
function update_partable!(partable::EnsembleParameterTable, model_identifier::AbstractDict, vec, column)
    for k in keys(partable.tables)
        update_partable!(partable.tables[k], model_identifier, vec, column)
    end
    return partable
end