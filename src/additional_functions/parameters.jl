function fill_A_S_M(A, S, M, A_indices, S_indices, M_indices, parameters)

    for (iA, iS, par) in zip(A_indices, S_indices, parameters)

        for index_A in iA
            A[index_A] = par
        end

        for index_S in iS
            S[index_S] = par
        end

    end

    if !isnothing(M)

        for (iM, par) in zip(M_indices, parameters)

            for index_M in iM
                M[index_M] = par
            end

        end

    end

end

function get_parameter_indices(parameters, M; linear = true, kwargs...)

    M_indices = [findall(x -> (x == par), M) for par in parameters]

    if linear
        M_indices = cartesian2linear.(M_indices, [M])
    end

    return M_indices

end

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

cartesian2linear(ind_cart, A::AbstractArray) = cartesian2linear(ind_cart, size(A))
linear2cartesian(ind_linear, A::AbstractArray) = linear2cartesian(ind_linear, size(A))

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


function get_matrix_derivative(M_indices, parameters, n_long)

    ∇M = [
    sparsevec(
        M_indices[i], 
        ones(length(M_indices[i])),
        n_long) for i in 1:length(parameters)]

    ∇M = hcat(∇M...)

    return ∇M

end

function fill_matrix(M, M_indices, parameters)

    for (iM, par) in zip(M_indices, parameters)

        for index_M in iM
            M[index_M] = par
        end

    end

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
            throw(ErrorException("Your parameter vector is not partitioned into directed and undirected effects"))
            return nothing
        end
    end

    for i in first_S:last_S
        if length(S_indices[i]) == 0
            throw(ErrorException("Your parameter vector is not partitioned into directed and undirected effects"))
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
            throw(ErrorException("Your parameter vector is not partitioned into directed, undirected and mean effects"))
            return nothing
        end
    end

    return first_M:last_M

end