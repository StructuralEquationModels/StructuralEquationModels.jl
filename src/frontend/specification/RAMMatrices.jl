
############################################################################################
### Type
############################################################################################

struct RAMMatrices <: SemSpecification
    A::ParamsMatrix{Float64}
    S::ParamsMatrix{Float64}
    F::SparseMatrixCSC{Float64}
    M::Union{ParamsVector{Float64}, Nothing}
    parameters::Vector{Symbol}
    colnames::Union{Vector{Symbol}, Nothing}    # better call it "variables": it's a mixture of observed and latent (and it gets confusing with get_colnames())
end

nparams(ram::RAMMatrices) = nparams(ram.A)
nvars(ram::RAMMatrices) = size(ram.F, 2)
nobserved_vars(ram::RAMMatrices) = size(ram.F, 1)
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

vars(ram::RAMMatrices) = ram.colnames

isobserved_var(ram::RAMMatrices, i::Integer) =
    ram.F.colptr[i+1] > ram.F.colptr[i]
islatent_var(ram::RAMMatrices, i::Integer) =
    ram.F.colptr[i+1] == ram.F.colptr[i]

observed_var_indices(ram::RAMMatrices) =
    [i for i in axes(ram.F, 2) if isobserved_var(ram, i)]
latent_var_indices(ram::RAMMatrices) =
    [i for i in axes(ram.F, 2) if islatent_var(ram, i)]

function observed_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return [col for (i, col) in enumerate(ram.colnames)
                if isobserved_var(ram, i)]
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return [col for (i, col) in enumerate(ram.colnames)
                if islatent_var(ram, i)]
    end
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(; A::AbstractMatrix, S::AbstractMatrix,
                       F::AbstractMatrix, M::Union{AbstractVector, Nothing} = nothing,
                     parameters::AbstractVector{Symbol},
                     colnames::Union{AbstractVector{Symbol}, Nothing} = nothing)
    ncols = size(A, 2)
    if !isnothing(colnames)
        length(colnames) == ncols || throw(DimensionMismatch("colnames length ($(length(colnames))) does not match the number of columns in A ($ncols)"))
    end
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(DimensionMismatch("A should have as many rows and columns as colnames length ($ncols), $(size(A)) found"))
    size(S, 2) == ncols || throw(DimensionMismatch("S should have as many rows and columns as colnames length ($ncols), $(size(S)) found"))
    size(F, 2) == ncols || throw(DimensionMismatch("F should have as many columns as colnames length ($ncols), $(size(F, 2)) found"))
    if !isnothing(M)
        length(M) == ncols || throw(DimensionMismatch("M should have as many elements as colnames length ($ncols), $(length(M)) found"))
    end
    A = ParamsMatrix{Float64}(A, parameters)
    S = ParamsMatrix{Float64}(S, parameters)
    M = !isnothing(M) ? ParamsVector{Float64}(M, parameters) : nothing
    spF = sparse(F)
    if any(!isone, spF.nzval)
        throw(ArgumentError("F should contain only 0s and 1s"))
    end
    return RAMMatrices(A, S, F, M, parameters, colnames)
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
    n_vars = n_observed + n_latent

    # colnames (variables)
    # and F indices (map from each observed column to its variable index)
    if length(partable.variables.sorted) != 0
        @assert length(partable.variables.sorted) == nvars(partable)
        colnames = copy(partable.variables.sorted)
        F_inds = findall(∈(Set(partable.variables.observed)),
                         colnames)
    else
        colnames = [partable.variables.observed;
                    partable.variables.latent]
        F_inds = 1:n_observed
    end

    # indices of the colnames
    cols_index = Dict(col => i for (i, col) in enumerate(colnames))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    T = nonmissingtype(eltype(partable.columns.value_fixed))
    A_inds = [Vector{Int64}() for _ in 1:length(params)]
    A_lin_ixs = LinearIndices((n_vars, n_vars))
    S_inds = [Vector{Int64}() for _ in 1:length(params)]
    S_lin_ixs = LinearIndices((n_vars, n_vars))
    A_consts = Vector{Pair{Int, T}}()
    S_consts = Vector{Pair{Int, T}}()
    # is there a meanstructure?
    M_inds = any(==(Symbol("1")), partable.columns.from) ?
        [Vector{Int64}() for _ in 1:length(params)] : nothing
    M_consts = !isnothing(M_inds) ? Vector{Pair{Int, T}}() : nothing

    for row in partable

        row_ind = cols_index[row.to]
        col_ind = row.from != Symbol("1") ? cols_index[row.from] : nothing

        if !row.free
            if (row.parameter_type == :→) && (row.from == Symbol("1"))
                push!(M_consts, row_ind => row.value_fixed)
            elseif (row.parameter_type == :→)
                push!(A_consts, A_lin_ixs[CartesianIndex(row_ind, col_ind)] => row.value_fixed)
            elseif (row.parameter_type == :↔)
                push!(S_consts, S_lin_ixs[CartesianIndex(row_ind, col_ind)] => row.value_fixed)
                if row_ind != col_ind # symmetric
                    push!(S_consts, S_lin_ixs[CartesianIndex(col_ind, row_ind)] => row.value_fixed)
                end
            else
                error("Unsupported parameter type: $(row.parameter_type)")
            end
        else
            par_ind = params_index[row.identifier]
            if (row.parameter_type == :→) && (row.from == Symbol("1"))
                push!(M_inds[par_ind], row_ind)
            elseif row.parameter_type == :→
                push!(A_inds[par_ind], A_lin_ixs[CartesianIndex(row_ind, col_ind)])
            elseif row.parameter_type == :↔
                push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(row_ind, col_ind)])
                if row_ind != col_ind # symmetric
                    push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(col_ind, row_ind)])
                end
            else
                error("Unsupported parameter type: $(row.parameter_type)")
            end
        end
    end
    # sort linear indices
    for A_ind in A_inds
        sort!(A_ind)
    end
    for S_ind in S_inds
        unique!(sort!(S_ind)) # also symmetric duplicates
    end
    if !isnothing(M_inds)
        for M_ind in M_inds
            sort!(M_ind)
        end
    end
    sort!(A_consts, by=first)
    sort!(S_consts, by=first)
    if !isnothing(M_consts)
        sort!(M_consts, by=first)
    end

    return RAMMatrices(ParamsMatrix{T}(A_inds, A_consts, (n_vars, n_vars)),
                       ParamsMatrix{T}(S_inds, S_consts, (n_vars, n_vars)),
                       sparse(1:n_observed, F_inds, ones(T, length(F_inds)), n_observed, n_vars),
                       !isnothing(M_inds) ? ParamsVector{T}(M_inds, M_consts, (n_vars,)) : nothing,
                       params, colnames)
end

Base.convert(::Type{RAMMatrices}, partable::ParameterTable) = RAMMatrices(partable)

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(ram_matrices::RAMMatrices)

    colnames = ram_matrices.colnames

    partable = ParameterTable(observed_vars = colnames[ram_matrices.F.rowval],
                              latent_vars = colnames[setdiff(eachindex(colnames),
                                                             ram_matrices.F.rowval)])

    position_names = Dict{Int, Symbol}(1:length(colnames) .=> colnames)

    append_rows!(partable, ram_matrices.A, :A,
                 ram_matrices.parameters, position_names)
    append_rows!(partable, ram_matrices.S, :S,
                 ram_matrices.parameters, position_names, skip_symmetric=true)
    if !isnothing(ram_matrices.M)
        append_rows!(partable, ram_matrices.M, :M,
                     ram_matrices.parameters, position_names)
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

function partable_row(val, index, matrix::Symbol,
                      position_names::AbstractDict;
                      free::Bool = true)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = position_names[index]
    else
        from = position_names[index[2]]
        to = position_names[index[1]]
    end

    return (
        from = from,
        parameter_type = matrix_to_parameter_type(matrix),
        to = to,
        free = free,
        value_fixed = free ? 0.0 : val,
        start = 0.0,
        estimate = 0.0,
        identifier = free ? val : :const)
end

function append_rows!(partable::ParameterTable,
                      arr::ParamsArray, arr_name::Symbol,
                      parameters::AbstractVector,
                      position_names;
                      skip_symmetric::Bool = false)
    nparams(arr) == length(params) ||
        throw(ArgumentError("Length of parameters vector does not match the number of parameters in the matrix"))
    arr_ixs = eachindex(arr)

    # add parameters
    visited_indices = Set{eltype(arr_ixs)}()
    for (i, par) in enumerate(parameters)
        for j in param_occurences_range(arr, i)
            arr_ix = arr_ixs[arr.linear_indices[j]]
            skip_symmetric && (arr_ix ∈ visited_indices) && continue

            push!(partable, partable_row(par, arr_ix, arr_name, position_names, free=true))
            if skip_symmetric
                # mark index and its symmetric as visited
                push!(visited_indices, arr_ix)
                push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
            end
        end
    end

    # add constants
    for (i, val) in arr.constants
        arr_ix = arr_ixs[i]
        skip_symmetric && (arr_ix ∈ visited_indices) && continue
        push!(partable, partable_row(val, arr_ix, arr_name, position_names, free=false))
        if skip_symmetric
            # mark index and its symmetric as visited
            push!(visited_indices, arr_ix)
            push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = ( (mat1.A == mat2.A) && (mat1.S == mat2.S) &&
            (mat1.F == mat2.F) && (mat1.M == mat2.M) &&
            (mat1.parameters == mat2.parameters) &&
            (mat1.colnames == mat2.colnames) )
    return res
end
