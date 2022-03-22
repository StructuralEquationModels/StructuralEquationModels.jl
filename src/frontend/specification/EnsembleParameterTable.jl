############################################################################
### Types
############################################################################

mutable struct EnsembleParameterTable{C}
    tables::C
end

############################################################################
### Constructors
############################################################################

# constuct an empty table
function EnsembleParameterTable(disambig::Nothing)
    tables = Dict{Symbol, ParameterTable}()
    return EnsembleParameterTable(tables)
end

############################################################################
### Convert to other types
############################################################################

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

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, partable::EnsembleParameterTable)
    print(io, "EnsembleParameterTable with groups: ")
    for key in keys(partable.tables) print(io, "|", key, "|") end
    print(io, "\n")
    for key in keys(partable.tables)
        print("\n")
        print(io, key, ": \n")
        print(partable.tables[key])
    end
end

#= function Base.show(io::IO, partable::EnsembleParameterTable)
    print(io, "EnsembleParameterTable with groups \n")
    print(partable.tables)
end =#

############################################################################
### Additional Methods
############################################################################

# Sorting -------------------------------------------------------------------

# todo

# add a row -------------------------------------------------------------------

# do we really need this?
import Base.push!

function push!(partable::EnsembleParameterTable, d::AbstractDict, group)
    push!(partable.tables[group], d)
end

push!(partable::EnsembleParameterTable, d::Nothing, group) = nothing

# get group -------------------------------------------------------------------

get_group(partable::EnsembleParameterTable, group) = get_group(partable.tables, group)

############################################################################
### Update Partable from Fitted Model
############################################################################

# ToDo