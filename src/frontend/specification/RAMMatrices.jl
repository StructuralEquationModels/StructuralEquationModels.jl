############################################################################################
### Constants
############################################################################################

struct RAMConstant
    matrix::Symbol
    index::CartesianIndex
    value
end

function Base.:(==)(c1::RAMConstant, c2::RAMConstant)
    res = ( (c1.matrix == c2.matrix) && (c1.index == c2.index) &&
            (c1.value == c2.value) )
    return res
end

function append_RAMConstants!(constants::AbstractVector{RAMConstant},
                              mtx_name::Symbol, mtx::AbstractArray;
                              skip_zeros::Bool = true)
    for (index, val) in pairs(mtx)
        if isa(val, Number) && !(skip_zeros && iszero(val))
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

struct RAMMatrices <: SemSpecification
    A_ind::ArrayParamsMap
    S_ind::ArrayParamsMap
    F_ind::Vector{Int}
    M_ind::Union{ArrayParamsMap, Nothing}
    parameters::Vector{Symbol}
    colnames::Vector{Symbol}
    constants::Vector{RAMConstant}
    size_F::Tuple{Int, Int}
end

nparams(ram::RAMMatrices) = length(ram.A_ind)
nvars(ram::RAMMatrices) = ram.size_F[2]
nobserved_vars(ram::RAMMatrices) = ram.size_F[1]
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

function observed_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return view(ram.colnames, ram.F_ind)
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return view(ram.colnames, setdiff(eachindex(ram.colnames), ram.F_ind))
    end
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(; A::AbstractMatrix, S::AbstractMatrix,
                       F::AbstractMatrix, M::Union{AbstractVector, Nothing} = nothing,
                     parameters::AbstractVector{Symbol},
                     colnames::AbstractVector{Symbol})
    ncols = length(colnames)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(DimensionMismatch("A should have as many rows and columns as colnames length ($(length(colnames))), $(size(A)) found"))
    size(S, 2) == ncols || throw(DimensionMismatch("S should have as many rows and columns as colnames length ($(length(colnames))), $(size(S)) found"))
    size(F, 2) == ncols || throw(DimensionMismatch("F should have as many columns as colnames length ($(length(colnames))), $(size(F, 2)) found"))
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

Base.convert(::Type{RAMMatrices}, partable::ParameterTable) = RAMMatrices(partable)

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
        push!(partable, partable_row(c, position_names))
    end

    # parameters
    for (i, par) in enumerate(ram_matrices.parameters)
        append_partable_rows!(
            partable, position_names,
            par, i,
            ram_matrices.A_ind,
            ram_matrices.S_ind,
            ram_matrices.M_ind,
            ram_matrices.size_F[2])
    end

    return partable
end

Base.convert(::Type{<:ParameterTable}, ram_matrices::RAMMatrices) = ParameterTable(ram_matrices)

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

function matrix_to_parameter_type(matrix::Symbol)
    if matrix == :A
        return :→
    elseif matrix == :S
        return :↔
    elseif matrix == :M
        return :→
    else
        throw(ArgumentError("Unsupported matrix $matrix, supported matrices are :A, :S and :M"))
    end
end

partable_row(c::RAMConstant, position_names::AbstractDict) = (
        from = position_names[c.index[2]],
        parameter_type = matrix_to_parameter_type(c.matrix),
        to = position_names[c.index[1]],
        free = false,
        value_fixed = c.value,
        start = 0.0,
        estimate = 0.0,
        identifier = :const
    )

function partable_row(par::Symbol, position_names::AbstractDict,
                      index::Integer, matrix::Symbol, n_nod::Integer)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = position_names[index]
    else
        cart_index = linear2cartesian(index, (n_nod, n_nod))

        from = position_names[cart_index[2]]
        to = position_names[cart_index[1]]
    end

    return (
        from = from,
        parameter_type = matrix_to_parameter_type(matrix),
        to = to,
        free = true,
        value_fixed = 0.0,
        start = 0.0,
        estimate = 0.0,
        identifier = par)
end

function append_partable_rows!(partable::ParameterTable,
                               position_names, par::Symbol, par_index::Integer,
                               A_ind, S_ind, M_ind, n_nod::Integer)
    for ind in A_ind[par_index]
        push!(partable, partable_row(par, position_names, ind, :A, n_nod))
    end

    visited_S_indices = Set{Int}()
    for ind in S_ind[par_index]
        if ind ∉ visited_S_indices
            push!(partable, partable_row(par, position_names, ind, :S, n_nod))
            # mark index and its symmetric as visited
            push!(visited_S_indices, ind)
            cart_index = linear2cartesian(ind, (n_nod, n_nod))
            push!(visited_S_indices, cartesian2linear(CartesianIndex(cart_index[2], cart_index[1]), (n_nod, n_nod)))
        end
    end

    if !isnothing(M_ind)
        for ind in M_ind[par_index]
            push!(partable, partable_row(par, position_names, ind, :M, n_nod))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = ( (mat1.A_ind == mat2.A_ind) && (mat1.S_ind == mat2.S_ind) &&
            (mat1.F_ind == mat2.F_ind) && (mat1.M_ind == mat2.M_ind) &&
            (mat1.parameters == mat2.parameters) &&
            (mat1.colnames == mat2.colnames) && (mat1.size_F == mat2.size_F) &&
            (mat1.constants == mat2.constants) )
    return res
end
