# fill A, S, and M matrices with the parameter values according to the parameters map
function fill_A_S_M!(
    A::AbstractMatrix, S::AbstractMatrix,
    M::Union{AbstractVector, Nothing},
    A_indices::AbstractArrayParamsMap,
    S_indices::AbstractArrayParamsMap,
    M_indices::Union{AbstractArrayParamsMap, Nothing},
    parameters::AbstractVector
)

    @inbounds for (iA, iS, par) in zip(A_indices, S_indices, parameters)
        for index_A in iA
            A[index_A] = par
        end

        for index_S in iS
            S[index_S] = par
        end

    end

    if !isnothing(M)

        @inbounds for (iM, par) in zip(M_indices, parameters)
            for index_M in iM
                M[index_M] = par
            end

        end

    end

end

# build the map from the index of the parameter to the indices of this parameter
# occurences in the given array
function array_parameters_map(parameters::AbstractVector, M::AbstractArray)
    params_index = Dict(param => i for (i, param) in enumerate(parameters))
    T = Base.eltype(eachindex(M))
    res = [Vector{T}() for _ in eachindex(parameters)]
    for (i, val) in pairs(M)
        par_ind = get(params_index, val, nothing)
        if !isnothing(par_ind)
            push!(res[par_ind], i)
        end
    end
    return res
end

# build the map of parameter index to the linear indices of its occurences in M
# returns ArrayParamsMap object
array_parameters_map_linear(parameters::AbstractVector, M::AbstractArray) =
    array_parameters_map(parameters, vec(M))

function eachindex_lower(M; linear_indices = false, kwargs...)

    indices = CartesianIndices(M)
    indices = filter(x -> (x[1] >= x[2]), indices)

    if linear_indices
        indices = cartesian2linear(indices, M)
    end

    return indices

end

function cartesian2linear(ind_cart, dims)
    ind_lin = LinearIndices(dims)[ind_cart]
    return ind_lin
end

function linear2cartesian(ind_lin, dims)
    ind_cart = CartesianIndices(dims)[ind_lin]
    return ind_cart
end

function set_constants!(M, M_pre)

    for index in eachindex(M)

        δ = tryparse(Float64, string(M[index]))

        if !iszero(M[index]) & (δ !== nothing)
            M_pre[index] = δ
        end

    end

end

function check_constants(M)

    for index in eachindex(M)

        δ = tryparse(Float64, string(M[index]))

        if !iszero(M[index]) & (δ !== nothing)
            return true
        end

    end

    return false

end


# construct length(M)×length(parameters) sparse matrix of 1s at the positions,
# where the corresponding parameter occurs in the M matrix
function matrix_gradient(M_indices::ArrayParamsMap,
                         M_length::Integer)
    rowval = reduce(vcat, M_indices)
    colptr = pushfirst!(accumulate((ptr, M_ind) -> ptr + length(M_ind), M_indices, init=1), 1)
    return SparseMatrixCSC(M_length, length(M_indices),
                colptr, rowval, ones(length(rowval)))
end

# fill M with parameters
function fill_matrix!(M::AbstractMatrix, M_indices::AbstractArrayParamsMap,
                      parameters::AbstractVector)

    for (iM, par) in zip(M_indices, parameters)
        for index_M in iM
            M[index_M] = par
        end
    end
    return M
end

function get_partition(A_indices, S_indices)

    n_par = length(A_indices)

    first_A = "a"
    first_S = "a"
    last_A = "a"
    last_S = "a"
    
    for i in 1:n_par
        if length(A_indices[i]) != 0
            first_A = i
            break
        end
    end

    for i in 1:n_par
        if length(S_indices[i]) != 0
            first_S = i
            break
        end
    end

    for i in n_par+1 .- (1:n_par)
        if length(A_indices[i]) != 0
            last_A = i
            break
        end
    end

    for i in n_par+1 .- (1:n_par)
        if length(S_indices[i]) != 0
            last_S = i
            break
        end
    end

    for i in first_A:last_A
        if length(A_indices[i]) == 0
            throw(ErrorException(
                "Your parameter vector is not partitioned into directed and undirected effects"))
            return nothing
        end
    end

    for i in first_S:last_S
        if length(S_indices[i]) == 0
            throw(ErrorException(
                "Your parameter vector is not partitioned into directed and undirected effects"))
            return nothing
        end
    end

    return first_A:last_A, first_S:last_S

end

function get_partition(M_indices)

    n_par = length(M_indices)

    first_M = "a"
    last_M = "a"
    
    for i in 1:n_par
        if length(M_indices[i]) != 0
            first_M = i
            break
        end
    end

    for i in n_par+1 .- (1:n_par)
        if length(M_indices[i]) != 0
            last_M = i
            break
        end
    end

    for i in first_M:last_M
        if length(M_indices[i]) == 0
            throw(ErrorException(
                "Your parameter vector is not partitioned into directed, undirected and mean effects"))
            return nothing
        end
    end

    return first_M:last_M

end