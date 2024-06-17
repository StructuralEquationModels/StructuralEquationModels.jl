"""
Array with partially parameterized elements.
"""
struct ParamsArray{T, N} <: AbstractArray{T, N}
    linear_indices::Vector{Int}     # linear indices of the parameter refs in the destination array
    nz_indices::Vector{Int}         # indices of the parameters refs in nonzero elements vector
    # (including the constants) ordered by the linear index
    param_ptr::Vector{Int}          # i-th element marks the start of the range in linear/nonzero
    # indices arrays that corresponds to the i-th parameter
    # (nparams + 1 elements)
    constants::Vector{Tuple{Int, Int, T}} # linear index, index in nonzero vector, value
    size::NTuple{N, Int}            # size of the destination array
end

ParamsVector{T} = ParamsArray{T, 1}
ParamsMatrix{T} = ParamsArray{T, 2}

function ParamsArray{T, N}(
    params_map::AbstractVector{<:AbstractVector{Int}},
    constants::Vector{Pair{Int, T}},
    size::NTuple{N, Int},
) where {T, N}
    params_ptr =
        pushfirst!(accumulate((ptr, inds) -> ptr + length(inds), params_map, init = 1), 1)
    param_lin_inds = reduce(vcat, params_map, init = Vector{Int}())
    nz_lin_inds = unique!(sort!([param_lin_inds; first.(constants)]))
    if length(nz_lin_inds) < length(param_lin_inds) + length(constants)
        throw(ArgumentError("Duplicate linear indices in the parameterized array"))
    end
    return ParamsArray{T, N}(
        param_lin_inds,
        searchsortedfirst.(Ref(nz_lin_inds), param_lin_inds),
        params_ptr,
        [(c[1], searchsortedfirst(nz_lin_inds, c[1]), c[2]) for c in constants],
        size,
    )
end

function ParamsArray{T, N}(
    arr::AbstractArray{<:Any, N},
    params::AbstractVector{Symbol};
    skip_zeros::Bool = true,
) where {T, N}
    params_index = Dict(param => i for (i, param) in enumerate(params))
    constants = Vector{Pair{Int, T}}()
    params_map = [Vector{Int}() for _ in eachindex(params)]
    arr_ixs = CartesianIndices(arr)
    for (i, val) in pairs(vec(arr))
        ismissing(val) && continue
        if isa(val, Number)
            (skip_zeros && iszero(val)) || push!(constants, i => val)
        else
            par_ind = get(params_index, val, nothing)
            if !isnothing(par_ind)
                push!(params_map[par_ind], i)
            else
                throw(KeyError("Unrecognized parameter $val at position $(arr_ixs[i])"))
            end
        end
    end
    return ParamsArray{T, N}(params_map, constants, size(arr))
end

ParamsArray{T}(
    arr::AbstractArray{<:Any, N},
    params::AbstractVector{Symbol};
    kwargs...,
) where {T, N} = ParamsArray{T, N}(arr, params; kwargs...)

nparams(arr::ParamsArray) = length(arr.param_ptr) - 1
SparseArrays.nnz(arr::ParamsArray) = length(arr.linear_indices) + length(arr.constants)

Base.size(arr::ParamsArray) = arr.size
Base.size(arr::ParamsArray, i::Integer) = arr.size[i]

Base.:(==)(a::ParamsArray, b::ParamsArray) = return eltype(a) == eltype(b) &&
       size(a) == size(b) &&
       a.constants == b.constants &&
       a.param_ptr == b.param_ptr &&
       a.linear_indices == b.linear_indices

# the range of arr.param_ptr indices that correspond to i-th parameter
param_occurences_range(arr::ParamsArray, i::Integer) =
    arr.param_ptr[i]:(arr.param_ptr[i+1]-1)

"""
    param_occurences(arr::ParamsArray, i::Integer)

Get the linear indices of the elements in `arr` that equal to the
`i`-th parameter.
"""
param_occurences(arr::ParamsArray, i::Integer) =
    view(arr.linear_indices, arr.param_ptr[i]:(arr.param_ptr[i+1]-1))

"""
    materialize!(dest::AbstractArray{<:Any, N}, src::ParamsArray{<:Any, N},
                 param_values::AbstractVector;
                 set_constants::Bool = true,
                 set_zeros::Bool = false)

Materialize the parameterized array `src` into `dest` by substituting the parameter
references with the parameter values from `param_values`.
"""
function materialize!(
    dest::AbstractArray{<:Any, N},
    src::ParamsArray{<:Any, N},
    param_values::AbstractVector;
    set_constants::Bool = true,
    set_zeros::Bool = false,
) where {N}
    size(dest) == size(src) || throw(
        DimensionMismatch(
            "Parameters ($(size(params_arr))) and destination ($(size(dest))) array sizes don't match",
        ),
    )
    nparams(src) == length(param_values) || throw(
        DimensionMismatch(
            "Number of values ($(length(param_values))) does not match the number of parameters ($(nparams(src)))",
        ),
    )
    Z = eltype(dest) <: Number ? eltype(dest) : eltype(src)
    set_zeros && fill!(dest, zero(Z))
    if set_constants
        @inbounds for (i, _, val) in src.constants
            dest[i] = val
        end
    end
    @inbounds for (i, val) in enumerate(param_values)
        for j in param_occurences_range(src, i)
            dest[src.linear_indices[j]] = val
        end
    end
    return dest
end

function materialize!(
    dest::SparseMatrixCSC,
    src::ParamsMatrix,
    param_values::AbstractVector;
    set_constants::Bool = true,
    set_zeros::Bool = false,
)
    set_zeros && throw(ArgumentError("Cannot set zeros for sparse matrix"))
    size(dest) == size(src) || throw(
        DimensionMismatch(
            "Parameters ($(size(params_arr))) and destination ($(size(dest))) array sizes don't match",
        ),
    )
    nparams(src) == length(param_values) || throw(
        DimensionMismatch(
            "Number of values ($(length(param_values))) does not match the number of parameters ($(nparams(src)))",
        ),
    )

    nnz(dest) == nnz(src) || throw(
        DimensionMismatch(
            "Number of non-zero elements ($(nnz(dest))) does not match the number of parameter references and constants ($(nnz(src)))",
        ),
    )
    if set_constants
        @inbounds for (_, j, val) in src.constants
            dest.nzval[j] = val
        end
    end
    @inbounds for (i, val) in enumerate(param_values)
        for j in param_occurences_range(src, i)
            dest.nzval[src.nz_indices[j]] = val
        end
    end
    return dest
end

"""
    materialize([T], src::ParamsArray{<:Any, N},
                param_values::AbstractVector{T}) where T

Materialize the parameterized array `src` into a new array of type `T`
by substituting the parameter references with the parameter values from `param_values`.
"""
materialize(::Type{T}, arr::ParamsArray, param_values::AbstractVector) where {T} =
    materialize!(similar(arr, T), arr, param_values, set_constants = true, set_zeros = true)

materialize(arr::ParamsArray, param_values::AbstractVector{T}) where {T} =
    materialize(Union{T, eltype(arr)}, arr, param_values)

# the hack to update the structured matrix (should be fine since the structure is imposed by ParamsMatrix)
materialize!(
    dest::Union{Symmetric, LowerTriangular, UpperTriangular},
    src::ParamsMatrix{<:Any},
    param_values::AbstractVector;
    kwargs...,
) = materialize!(parent(dest), src, param_values; kwargs...)

function sparse_materialize(
    ::Type{T},
    arr::ParamsMatrix,
    param_values::AbstractVector,
) where {T}
    nparams(arr) == length(param_values) || throw(
        DimensionMismatch(
            "Number of values ($(length(param_values))) does not match the number of parameter ($(nparams(arr)))",
        ),
    )

    nz_vals = Vector{T}(undef, nnz(arr))
    nz_lininds = Vector{Int}(undef, nnz(arr))
    # fill constants
    @inbounds for (lin_ind, nz_ind, val) in arr.constants
        nz_vals[nz_ind] = val
        nz_lininds[nz_ind] = lin_ind
    end
    # fill parameters
    @inbounds for (i, val) in enumerate(param_values)
        for j in param_occurences_range(arr, i)
            nz_ind = arr.nz_indices[j]
            nz_vals[nz_ind] = val
            nz_lininds[nz_ind] = arr.linear_indices[j]
        end
    end
    arr_ixs = CartesianIndices(size(arr))
    return sparse(
        [arr_ixs[i][1] for i in nz_lininds],
        [arr_ixs[i][2] for i in nz_lininds],
        nz_vals,
        size(arr)...,
    )
end

sparse_materialize(arr::ParamsArray, params::AbstractVector{T}) where {T} =
    sparse_materialize(Union{T, eltype(arr)}, arr, params)

# construct length(M)×length(params) sparse matrix of 1s at the positions,
# where the corresponding parameter occurs in the arr
sparse_gradient(::Type{T}, arr::ParamsArray) where {T} = SparseMatrixCSC(
    length(arr),
    nparams(arr),
    arr.param_ptr,
    arr.linear_indices,
    ones(T, length(arr.linear_indices)),
)

sparse_gradient(arr::ParamsArray{T}) where {T} = sparse_gradient(T, arr)

# range of parameters that are referenced in the matrix
function params_range(arr::ParamsArray; allow_gaps::Bool = false)
    first_i = findfirst(i -> arr.param_ptr[i+1] > arr.param_ptr[i], 1:nparams(arr)-1)
    last_i = findlast(i -> arr.param_ptr[i+1] > arr.param_ptr[i], 1:nparams(arr)-1)

    if !allow_gaps && !isnothing(first_i) && !isnothing(last_i)
        for i in first_i:last_i
            if isempty(param_occurences_range(arr, i))
                # TODO show which parameter is missing in which matrix
                throw(
                    ErrorException(
                        "Parameter vector is not partitioned into directed and undirected effects",
                    ),
                )
            end
        end
    end

    return first_i:last_i
end
