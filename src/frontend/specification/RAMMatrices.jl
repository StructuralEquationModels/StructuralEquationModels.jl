############################################################################################
### Constants
############################################################################################

struct RAMConstant
    matrix::Symbol
    index::CartesianIndex
    value
end

import Base.==

function ==(c1::RAMConstant, c2::RAMConstant)
    res = ( (c1.matrix == c2.matrix) && (c1.index == c2.index) &&
            (c1.value == c2.value) )
    return res
end

function append_RAMConstants!(constants::AbstractVector{RAMConstant},
                              mtx_name::Symbol, mtx::AbstractArray)
    for (index, val) in pairs(mtx)
        if isa(val, Number) && !iszero(val)
            push!(constants, RAMConstant(mtx_name, index, val))
        end
    end
    return constants
end

function set_RAMConstant!(A, S, M, rc::RAMConstant)
    if rc.matrix == :A
        A[rc.index] = rc.value
    elseif rc.matrix == :S
        S[rc.index] = rc.value
        S[rc.index[2], rc.index[1]] = rc.value # symmetric
    elseif rc.matrix == :M
        M[rc.index] = rc.value
    end
end

function set_RAMConstants!(A, S, M, rc_vec::Vector{RAMConstant})
    for rc in rc_vec set_RAMConstant!(A, S, M, rc) end
end

############################################################################################
### Type
############################################################################################

# map from parameter index to linear indices of matrix/vector positions where it occurs
AbstractArrayParamsMap = AbstractVector{<:AbstractVector{<:Integer}}
ArrayParamsMap = Vector{Vector{Int}}

struct RAMMatrices
    A_ind::ArrayParamsMap
    S_ind::ArrayParamsMap
    F_ind::Vector{Int}
    M_ind::Union{ArrayParamsMap, Nothing}
    parameters
    colnames
    constants::Vector{RAMConstant}
    size_F
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(;A, S, F, M = nothing, parameters, colnames)
    A_indices = array_parameters_map_linear(parameters, A)
    S_indices = array_parameters_map_linear(parameters, S)
    M_indices = !isnothing(M) ? array_parameters_map_linear(parameters, M) : nothing
    F_indices = [i for (i, col) in zip(axes(F, 2), eachcol(F)) if any(isone, col)]
    constants = Vector{RAMConstant}()
    append_RAMConstants!(constants, :A, A)
    append_RAMConstants!(constants, :S, S)
    isnothing(M) || append_RAMConstants!(constants, :M, M)
    return RAMMatrices(A_indices, S_indices, F_indices, M_indices,
                       parameters, colnames, constants, size(F))
end

RAMMatrices(a::RAMMatrices) = a

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(partable::ParameterTable;
                     params::Union{AbstractVector{Symbol}, Nothing} = nothing)

    if isnothing(params)
        params = parameters(partable)
    end
    params_index = Dict(param => i for (i, param) in enumerate(params))
    if length(params) != length(params_index)
        params_seen = Set{Symbol}()
        params_nonunique = Vector{Symbol}()
        for par in params
            push!(par in params_seen ? params_nonunique : params_seen, par)
        end
        throw(ArgumentError("Duplicate names in the parameters vector: $(join(params_nonunique, ", "))"))
    end

    n_observed = length(partable.variables.observed)
    n_latent = length(partable.variables.latent)
    n_node = n_observed + n_latent

    # F indices
    F_ind = length(partable.variables.sorted) != 0 ?
        findall(∈(Set(partable.variables.observed)),
                partable.variables.sorted) :
        1:n_observed

    # indices of the colnames
    colnames = length(partable.variables.sorted) != 0 ?
        copy(partable.variables.sorted) :
        [partable.variables.observed;
         partable.variables.latent]
    cols_index = Dict(col => i for (i, col) in enumerate(colnames))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    A_ind = [Vector{Int64}() for _ in 1:length(params)]
    S_ind = [Vector{Int64}() for _ in 1:length(params)]

    # is there a meanstructure?
    M_ind = any(==(Symbol("1")), partable.columns.from) ?
        [Vector{Int64}() for _ in 1:length(params)] : nothing

    # handle constants
    constants = Vector{RAMConstant}()

    for row in partable

        row_ind = cols_index[row.to]
        col_ind = row.from != Symbol("1") ? cols_index[row.from] : nothing

        if !row.free
            if (row.parameter_type == :→) && (row.from == Symbol("1"))
                push!(constants, RAMConstant(:M, row_ind, row.value_fixed))
            elseif (row.parameter_type == :→)
                push!(constants, RAMConstant(:A, CartesianIndex(row_ind, col_ind), row.value_fixed))
            elseif (row.parameter_type == :↔)
                push!(constants, RAMConstant(:S, CartesianIndex(row_ind, col_ind), row.value_fixed))
            else
                error("Unsupported parameter type: $(row.parameter_type)")
            end
        else
            par_ind = params_index[row.identifier]
            if (row.parameter_type == :→) && (row.from == Symbol("1"))
                push!(M_ind[par_ind], row_ind)
            elseif row.parameter_type == :→
                push!(A_ind[par_ind], row_ind + (col_ind-1)*n_node)
            elseif row.parameter_type == :↔
                push!(S_ind[par_ind], row_ind + (col_ind-1)*n_node)
                if row_ind != col_ind
                    push!(S_ind[par_ind], col_ind + (row_ind-1)*n_node)
                end
            else
                error("Unsupported parameter type: $(row.parameter_type)")
            end
        end

    end

    return RAMMatrices(A_ind, S_ind, F_ind, M_ind,
                       params, colnames, constants,
                       (n_observed, n_node))
end

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(ram_matrices::RAMMatrices)

    colnames = ram_matrices.colnames

    partable = ParameterTable(observed_vars = colnames[ram_matrices.F_ind],
                              latent_vars = colnames[setdiff(eachindex(colnames),
                                                             ram_matrices.F_ind)])

    position_names = Dict{Int64, Symbol}(1:length(colnames) .=> colnames)

    # constants
    for c in ram_matrices.constants
        push!(partable, get_partable_row(c, position_names))
    end

    # parameters
    for (i, par) in enumerate(ram_matrices.parameters)
        push_partable_rows!(
            partable, position_names,
            par, i,
            ram_matrices.A_ind,
            ram_matrices.S_ind,
            ram_matrices.M_ind,
            ram_matrices.size_F[2])
    end

    return partable
end


############################################################################################
### get RAMMatrices from EnsembleParameterTable
############################################################################################

function RAMMatrices(partable::EnsembleParameterTable)

    ram_matrices = Dict{Symbol, RAMMatrices}()

    params = parameters(partable)

    for key in keys(partable.tables)
        ram_mat = RAMMatrices(partable.tables[key]; params = params)
        push!(ram_matrices, key => ram_mat)
    end

    return ram_matrices
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end

############################################################################################
### Additional Functions
############################################################################################

# get the vector of all parameters in the table
# the position of the parameter is based on its first appearance in the table (and the ensemble)
function parameters(partable::Union{EnsembleParameterTable, ParameterTable})
    if partable isa ParameterTable
        parameters = partable.columns.identifier
    else
        parameters = Vector{Symbol}()
        for tbl in values(partable.tables)
            append!(parameters, tbl.columns.identifier)
        end
    end
    parameters = unique(parameters)
    filter!(!=(:const), parameters) # exclude constants

    return parameters
end

function get_partable_row(c::RAMConstant, position_names)
    # variable names
    from = position_names[c.index[2]]
    to = position_names[c.index[1]]
    # parameter type
    if c.matrix == :A 
        parameter_type = :→
    elseif c.matrix == :S
        parameter_type = :↔
    elseif c.matrix == :M
        parameter_type = :→
    end
    free = false
    value_fixed = c.value
    start = 0.0
    estimate = 0.0
    identifier = :const
    return Dict(
        :from => from, 
        :parameter_type => parameter_type, 
        :to => to, 
        :free => free, 
        :value_fixed => value_fixed, 
        :start => start, 
        :estimate => estimate, 
        :identifier => identifier)
end

function cartesian_is_known(index, known_indices)
    known = false
    for k_in in known_indices
        if (index == k_in) | ((index[1] == k_in[2]) & (index[2] == k_in[1]))
            known = true
        end
    end
    return known
end

cartesian_is_known(index, known_indices::Nothing) = false

function get_partable_row(par, position_names, index, matrix, n_nod, known_indices)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = position_names[index]
    else
        index = linear2cartesian(index, (n_nod, n_nod))

        if (matrix == :S) & (cartesian_is_known(index, known_indices))
            return nothing
        elseif matrix == :S
            push!(known_indices, index)
        end

        from = position_names[index[2]]
        to = position_names[index[1]]
    end

    # parameter type
    if matrix == :A 
        parameter_type = :→
    elseif matrix == :S
        parameter_type = :↔
    elseif matrix == :M
        parameter_type = :→
    end

    free = true
    value_fixed = 0.0
    start = 0.0
    estimate = 0.0
    identifier = par

    return Dict(
        :from => from, 
        :parameter_type => parameter_type, 
        :to => to, 
        :free => free, 
        :value_fixed => value_fixed,
        :start => start, 
        :estimate => estimate, 
        :identifier => identifier)
end

function push_partable_rows!(partable, position_names, par, i, A_ind, S_ind, M_ind, n_nod)
    A_ind = A_ind[i]
    S_ind = S_ind[i]
    isnothing(M_ind) || (M_ind = M_ind[i])

    for ind in A_ind
        push!(partable, get_partable_row(par, position_names, ind, :A, n_nod, nothing))
    end

    known_indices = Vector{CartesianIndex}()
    for ind in S_ind
        push!(partable, get_partable_row(par, position_names, ind, :S, n_nod, known_indices))
    end

    if !isnothing(M_ind)
        for ind in M_ind
            push!(partable, get_partable_row(par, position_names, ind, :M, n_nod, nothing))
        end
    end

    return nothing

end

function ==(mat1::RAMMatrices, mat2::RAMMatrices)
    res = ( (mat1.A_ind == mat2.A_ind) && (mat1.S_ind == mat2.S_ind) &&
            (mat1.F_ind == mat2.F_ind) && (mat1.M_ind == mat2.M_ind) &&
            (mat1.parameters == mat2.parameters) &&
            (mat1.colnames == mat2.colnames) && (mat1.size_F == mat2.size_F) &&
            (mat1.constants == mat2.constants) )
    return res
end

function get_group(d::Dict, group)
    return d[group]
end
