# Neumann series representation of (I - mat)⁻¹
function neumann_series(mat::SparseMatrixCSC; maxiter::Integer = size(mat, 1))
    inverse = I + mat
    next_term = mat^2

    n = 1
    while nnz(next_term) != 0
        (n <= maxiter) || error("Neumann series did not converge in $maxiter steps")
        inverse += next_term
        next_term *= mat
        n += 1
    end

    return inverse
end

function batch_inv!(fun, model)
    for i in 1:size(fun.inverses, 1)
        fun.inverses[i] .= LinearAlgebra.inv!(fun.choleskys[i])
    end
end

# computes A*S*B -> C, where ind gives the entries of S that are 1
function sparse_outer_mul!(C, A, B, ind)
    fill!(C, 0.0)
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[ind[i][2], :], C)
    end
end

# computes A*∇m, where ∇m ind gives the entries of ∇m that are 1
function sparse_outer_mul!(C, A, ind)
    fill!(C, 0.0)
    @views C .= sum(A[:, ind], dims = 2)
    return C
end

# computes A*S*B -> C, where ind gives the entries of S that are 1
function sparse_outer_mul!(C, A, B::Vector, ind)
    fill!(C, 0.0)
    @views @inbounds for i in 1:length(ind)
        C .+= B[ind[i][2]] .* A[:, ind[i][1]]
    end
end

# n²×(n(n+1)/2) matrix to transform a vector of lower
# triangular entries into a vectorized form of a n×n symmetric matrix,
# opposite of elimination_matrix()
function duplication_matrix(n::Integer)
    ntri = div(n * (n + 1), 2)
    D = zeros(n^2, ntri)
    for j in 1:n
        for i in j:n
            tri_ix               = (j - 1) * n + i - div(j * (j - 1), 2)
            D[j+n*(i-1), tri_ix] = 1
            D[i+n*(j-1), tri_ix] = 1
        end
    end
    return D
end

# (n(n+1)/2)×n² matrix to transform a
# vectorized form of a n×n symmetric matrix
# into vector of its lower triangular entries,
# opposite of duplication_matrix()
function elimination_matrix(n::Integer)
    ntri = div(n * (n + 1), 2)
    L = zeros(ntri, n^2)
    for j in 1:n
        for i in j:n
            tri_ix = (j - 1) * n + i - div(j * (j - 1), 2)
            L[tri_ix, i+n*(j-1)] = 1
        end
    end
    return L
end

# returns the vector of non-unique values in the order of appearance
# each non-unique values is reported once
function nonunique(values::AbstractVector)
    value_counts = Dict{eltype(values), Int}()
    res = similar(values, 0)
    for v in values
        n = get!(value_counts, v, 0)
        if n == 1 # second encounter
            push!(res, v)
        end
        value_counts[v] = n + 1
    end
    return res
end
