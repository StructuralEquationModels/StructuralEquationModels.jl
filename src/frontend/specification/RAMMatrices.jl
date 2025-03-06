
############################################################################################
### Type
############################################################################################

struct RAMMatrices <: SemSpecification
    A::param_labelsMatrix{Float64}
    S::param_labelsMatrix{Float64}
    F::SparseMatrixCSC{Float64}
    M::Union{param_labelsVector{Float64}, Nothing}
    param_labels::Vector{Symbol}
    vars::Union{Vector{Symbol}, Nothing}    # better call it "variables": it's a mixture of observed and latent (and it gets confusing with get_vars())
end

nparams(ram::RAMMatrices) = nparams(ram.A)
nvars(ram::RAMMatrices) = size(ram.F, 2)
nobserved_vars(ram::RAMMatrices) = size(ram.F, 1)
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

vars(ram::RAMMatrices) = ram.vars

isobserved_var(ram::RAMMatrices, i::Integer) = ram.F.colptr[i+1] > ram.F.colptr[i]
islatent_var(ram::RAMMatrices, i::Integer) = ram.F.colptr[i+1] == ram.F.colptr[i]

# indices of observed variables in the order as they appear in ram.F rows
function observed_var_indices(ram::RAMMatrices)
    obs_inds = Vector{Int}(undef, nobserved_vars(ram))
    @inbounds for i in 1:nvars(ram)
        colptr = ram.F.colptr[i]
        if ram.F.colptr[i+1] > colptr # is observed
            obs_inds[ram.F.rowval[colptr]] = i
        end
    end
    return obs_inds
end

latent_var_indices(ram::RAMMatrices) = [i for i in axes(ram.F, 2) if islatent_var(ram, i)]

# observed variables in the order as they appear in ram.F rows
function observed_vars(ram::RAMMatrices)
    if isnothing(ram.vars)
        @warn "Your RAMMatrices do not contain variable names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        obs_vars = Vector{Symbol}(undef, nobserved_vars(ram))
        @inbounds for (i, v) in enumerate(vars(ram))
            colptr = ram.F.colptr[i]
            if ram.F.colptr[i+1] > colptr # is observed
                obs_vars[ram.F.rowval[colptr]] = v
            end
        end
        return obs_vars
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.vars)
        @warn "Your RAMMatrices do not contain variable names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return [col for (i, col) in enumerate(ram.vars) if islatent_var(ram, i)]
    end
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(;
    A::AbstractMatrix,
    S::AbstractMatrix,
    F::AbstractMatrix,
    M::Union{AbstractVector, Nothing} = nothing,
    param_labels::AbstractVector{Symbol},
    vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    ncols = size(A, 2)
    isnothing(vars) || check_vars(vars, ncols)

    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(
        DimensionMismatch(
            "A should have as many rows and columns as vars length ($ncols), $(size(A)) found",
        ),
    )
    size(S, 2) == ncols || throw(
        DimensionMismatch(
            "S should have as many rows and columns as vars length ($ncols), $(size(S)) found",
        ),
    )
    size(F, 2) == ncols || throw(
        DimensionMismatch(
            "F should have as many columns as vars length ($ncols), $(size(F, 2)) found",
        ),
    )
    if !isnothing(M)
        length(M) == ncols || throw(
            DimensionMismatch(
                "M should have as many elements as vars length ($ncols), $(length(M)) found",
            ),
        )
    end
    check_param_labels(param_labels, nothing)

    A = param_labelsMatrix{Float64}(A, param_labels)
    S = param_labelsMatrix{Float64}(S, param_labels)
    M = !isnothing(M) ? param_labelsVector{Float64}(M, param_labels) : nothing
    spF = sparse(F)
    if any(!isone, spF.nzval)
        throw(ArgumentError("F should contain only 0s and 1s"))
    end
    return RAMMatrices(A, S, F, M, copy(param_labels), vars)
end

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(
    partable::ParameterTable;
    param_labels::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    param_labels = copy(isnothing(param_labels) ? SEM.param_labels(partable) : param_labels)
    check_param_labels(param_labels, partable.columns[:param])
    param_labels_index = param_indices(partable, nothing)

    n_observed = length(partable.observed_vars)
    n_latent = length(partable.latent_vars)
    n_vars = n_observed + n_latent

    if length(partable.sorted_vars) != 0
        @assert length(partable.sorted_vars) == nvars(partable)
        vars_sorted = copy(partable.sorted_vars)
    else
        vars_sorted = [
            partable.observed_vars
            partable.latent_vars
        ]
    end

    # indices of the vars (A/S/M rows or columns)
    vars_index = Dict(col => i for (i, col) in enumerate(vars_sorted))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    T = nonmissingtype(eltype(partable.columns[:value_fixed]))
    A_inds = [Vector{Int64}() for _ in 1:length(param_labels)]
    A_lin_ixs = LinearIndices((n_vars, n_vars))
    S_inds = [Vector{Int64}() for _ in 1:length(param_labels)]
    S_lin_ixs = LinearIndices((n_vars, n_vars))
    A_consts = Vector{Pair{Int, T}}()
    S_consts = Vector{Pair{Int, T}}()
    # is there a meanstructure?
    M_inds =
        any(==(Symbol(1)), partable.columns[:from]) ?
        [Vector{Int64}() for _ in 1:length(param_labels)] : nothing
    M_consts = !isnothing(M_inds) ? Vector{Pair{Int, T}}() : nothing

    for r in partable
        row_ind = vars_index[r.to]
        col_ind = r.from != Symbol(1) ? vars_index[r.from] : nothing

        if !r.free
            if (r.relation == :→) && (r.from == Symbol(1))
                push!(M_consts, row_ind => r.value_fixed)
            elseif r.relation == :→
                push!(
                    A_consts,
                    A_lin_ixs[CartesianIndex(row_ind, col_ind)] => r.value_fixed,
                )
            elseif r.relation == :↔
                push!(
                    S_consts,
                    S_lin_ixs[CartesianIndex(row_ind, col_ind)] => r.value_fixed,
                )
                if row_ind != col_ind # symmetric
                    push!(
                        S_consts,
                        S_lin_ixs[CartesianIndex(col_ind, row_ind)] => r.value_fixed,
                    )
                end
            else
                error("Unsupported relation: $(r.relation)")
            end
        else
            par_ind = param_labels_index[r.param]
            if (r.relation == :→) && (r.from == Symbol(1))
                push!(M_inds[par_ind], row_ind)
            elseif r.relation == :→
                push!(A_inds[par_ind], A_lin_ixs[CartesianIndex(row_ind, col_ind)])
            elseif r.relation == :↔
                push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(row_ind, col_ind)])
                if row_ind != col_ind # symmetric
                    push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(col_ind, row_ind)])
                end
            else
                error("Unsupported relation: $(r.relation)")
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
    sort!(A_consts, by = first)
    sort!(S_consts, by = first)
    if !isnothing(M_consts)
        sort!(M_consts, by = first)
    end

    return RAMMatrices(
        param_labelsMatrix{T}(A_inds, A_consts, (n_vars, n_vars)),
        param_labelsMatrix{T}(S_inds, S_consts, (n_vars, n_vars)),
        sparse(
            1:n_observed,
            [vars_index[var] for var in partable.observed_vars],
            ones(T, n_observed),
            n_observed,
            n_vars,
        ),
        !isnothing(M_inds) ? param_labelsVector{T}(M_inds, M_consts, (n_vars,)) : nothing,
        param_labels,
        vars_sorted,
    )
end

Base.convert(
    ::Type{RAMMatrices},
    partable::ParameterTable;
    param_labels::Union{AbstractVector{Symbol}, Nothing} = nothing,
) = RAMMatrices(partable; param_labels)

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(
    ram::RAMMatrices;
    param_labels::Union{AbstractVector{Symbol}, Nothing} = nothing,
    observed_var_prefix::Symbol = :obs,
    latent_var_prefix::Symbol = :var,
)
    # defer parameter checks until we know which ones are used

    if !isnothing(ram.vars)
        latent_vars = SEM.latent_vars(ram)
        observed_vars = SEM.observed_vars(ram)
        vars = ram.vars
    else
        observed_vars = [Symbol("$(observed_var_prefix)_$i") for i in 1:nobserved_vars(ram)]
        latent_vars = [Symbol("$(latent_var_prefix)_$i") for i in 1:nlatent_vars(ram)]
        vars = vcat(observed_vars, latent_vars)
    end

    # construct an empty table
    partable = ParameterTable(
        observed_vars = observed_vars,
        latent_vars = latent_vars,
        param_labels = isnothing(param_labels) ? SEM.param_labels(ram) : param_labels,
    )

    # fill the table
    append_rows!(partable, ram.S, :S, ram.param_labels, vars, skip_symmetric = true)
    append_rows!(partable, ram.A, :A, ram.param_labels, vars)
    if !isnothing(ram.M)
        append_rows!(partable, ram.M, :M, ram.param_labels, vars)
    end

    check_param_labels(SEM.param_labels(partable), partable.columns[:param])

    return partable
end

Base.convert(
    ::Type{<:ParameterTable},
    ram::RAMMatrices;
    param_labels::Union{AbstractVector{Symbol}, Nothing} = nothing,
) = ParameterTable(ram; param_labels)

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

# return the `from □ to` variables relation symbol (□) given the name of the source RAM matrix
function matrix_to_relation(matrix::Symbol)
    if matrix == :A
        return :→
    elseif matrix == :S
        return :↔
    elseif matrix == :M
        return :→
    else
        throw(
            ArgumentError(
                "Unsupported matrix $matrix, supported matrices are :A, :S and :M",
            ),
        )
    end
end

# generates a ParTable row NamedTuple for a given element of RAM matrix
function partable_row(
    val,
    index,
    matrix::Symbol,
    varnames::AbstractVector{Symbol};
    free::Bool = true,
)

    # variable names
    if matrix == :M
        from = Symbol(1)
        to = varnames[index]
    else
        from = varnames[index[2]]
        to = varnames[index[1]]
    end

    return (
        from = from,
        relation = matrix_to_relation(matrix),
        to = to,
        free = free,
        value_fixed = free ? 0.0 : val,
        start = 0.0,
        estimate = 0.0,
        param = free ? val : :const,
    )
end

function append_rows!(
    partable::ParameterTable,
    arr::param_labelsArray,
    arr_name::Symbol,
    param_labels::AbstractVector,
    varnames::AbstractVector{Symbol};
    skip_symmetric::Bool = false,
)
    nparams(arr) == length(param_labels) || throw(
        ArgumentError(
            "Length of parameters vector ($(length(param_labels))) does not match the number of parameters in the matrix ($(nparams(arr)))",
        ),
    )
    arr_ixs = eachindex(arr)

    # add parameters
    visited_indices = Set{eltype(arr_ixs)}()
    for (i, par) in enumerate(param_labels)
        for j in param_occurences_range(arr, i)
            arr_ix = arr_ixs[arr.linear_indices[j]]
            skip_symmetric && (arr_ix ∈ visited_indices) && continue

            push!(partable, partable_row(par, arr_ix, arr_name, varnames, free = true))
            if skip_symmetric
                # mark index and its symmetric as visited
                push!(visited_indices, arr_ix)
                push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
            end
        end
    end

    # add constants
    for (i, _, val) in arr.constants
        arr_ix = arr_ixs[i]
        skip_symmetric && (arr_ix ∈ visited_indices) && continue
        push!(partable, partable_row(val, arr_ix, arr_name, varnames, free = false))
        if skip_symmetric
            # mark index and its symmetric as visited
            push!(visited_indices, arr_ix)
            push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = (
        (mat1.A == mat2.A) &&
        (mat1.S == mat2.S) &&
        (mat1.F == mat2.F) &&
        (mat1.M == mat2.M) &&
        (mat1.param_labels == mat2.param_labels) &&
        (mat1.vars == mat2.vars)
    )
    return res
end
