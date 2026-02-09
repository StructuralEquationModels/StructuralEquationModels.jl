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

# truncate eigenvalues of a symmetric matrix and return the result
function trunc_eigvals(
    mtx::AbstractMatrix{T},
    min_eigval::Number;
    mtx_label::AbstractString = "matrix",
    verbose::Bool = false,
) where {T}
    # eigen decomposition of the mtx
    mtx_eig = eigen(convert(Matrix{T}, mtx))
    verbose &&
        @info "min(eigvals($mtx_label))=$(Base.minimum(mtx_eig.values)), N(eigvals < $min_eigval) = $(sum(<(min_eigval), mtx_eig.values))"

    eigmin = Base.minimum(mtx_eig.values)
    if eigmin < min_eigval
        # substitute small eigvals with min_eigval
        eigvals_mtx = Diagonal(max.(mtx_eig.values, min_eigval))
        newmtx = mtx_eig.vectors' * eigvals_mtx * mtx_eig.vectors
        StatsBase._symmetrize!(newmtx)
        if verbose
            Δmtx = newmtx .- mtx
            @info "Δ($mtx_label, posdef)=$(norm(Δmtx, 2)), min,max(Δᵢ)=$(extrema(Δmtx))"
        end
        return newmtx
    else
        return mtx
    end
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
