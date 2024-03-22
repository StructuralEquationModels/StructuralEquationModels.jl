# Neumann seriess representation of (I - mat)⁻¹
function neumann_series(mat::SparseMatrixCSC; maxn::Integer = size(mat, 1))
    inverse = I + mat
    next_term = mat^2

    n = 1
    while nnz(next_term) != 0
        (n <= maxn) || error("Neumann series did not converge in $maxn steps")
        inverse += next_term
        next_term *= mat
        n += 1
    end

    return inverse
end

#=
function make_onelement_array(A)
    isa(A, Array) ? nothing : (A = [A])
    return A
end
 =#

function semvec(observed, imply, loss, optimizer)

    observed = make_onelement_array(observed)
    imply = make_onelement_array(imply)
    loss = make_onelement_array(loss)
    optimizer = make_onelement_array(optimizer)

    #sem_vec = Array{AbstractSem}(undef, maximum(length.([observed, imply, loss, optimizer])))
    sem_vec = Sem.(observed, imply, loss, optimizer)

    return sem_vec
end

# construct a vector of SemObserved objects
# for each specified data row
function observed(::Type{T}, data, rowinds;
            args = (),
            kwargs = NamedTuple()) where T <: SemObserved
    return T[
        T(args...;
          data = Matrix(view(data, row, :)),
          kwargs...)
        for row in rowinds]
end

function skipmissing_mean(mat::AbstractMatrix)
    means = [mean(skipmissing(coldata))
             for coldata in eachcol(mat)]
    return means
end

function F_one_person(imp_mean, meandiff, inverse, data, logdet)
    F = logdet
    @. meandiff = data - imp_mean
    F += dot(meandiff, inverse, meandiff)
    return F
end

function remove_all_missing(data::AbstractMatrix)
    keep = Vector{Int64}()
    for (i, coldata) in zip(axes(data, 1), eachrow(data))
        if any(!ismissing, coldata)
            push!(keep, i)
        end
    end
    return data[keep, :], keep
end

#=
function batch_sym_inv_update!(fun::Union{LossFunction, DiffFunction}, model)
    M_inv = inv(fun.choleskys[1])
    for i = 1:size(fun.inverses, 1)
        if size(model.observed.patterns_not[i]) == 0
            fun.inverses[i] .= M_inv
        else
            ind_not = model.observed.patterns_not[i]
            ind = model.observed.patterns[i]

            A = M_inv[ind_not, ind]
            H = cholesky(M_inv[ind_not, ind_not])
            D = H \ A
            out = M_inv[ind, ind] - LinearAlgebra.BLAS.gemm('T', 'N', 1.0, A, D)
            fun.inverses[i] .= out
        end
    end
end =#

function sparse_outer_mul!(C, A, B, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[ind[i][2], :], C)
    end
end

function sparse_outer_mul!(C, A, ind) #computes A*∇m, where ∇m ind gives the entries of ∇m that are 1
    fill!(C, 0.0)
    @views C .= sum(A[:, ind], dims = 2)
    return C
end

function sparse_outer_mul!(C, A, B::Vector, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    @views @inbounds for i in 1:length(ind)
        C .+= B[ind[i][2]].*A[:, ind[i][1]]
    end
end

function duplication_matrix(nobs)
    nobs = Int(nobs)
    n1 = Int(nobs*(nobs+1)*0.5)
    n2 = Int(nobs^2)
    Dt = zeros(n1, n2)

    for j in 1:nobs
        for i in j:nobs
            u = zeros(n1)
            u[Int((j-1)*nobs + i-0.5*j*(j-1))] = 1
            T = zeros(nobs, nobs)
            T[j,i] = 1; T[i, j] = 1
            Dt += u*transpose(vec(T))
        end
    end
    D = transpose(Dt)
    return D
end

function elimination_matrix(nobs)
    nobs = Int(nobs)
    n1 = Int(nobs*(nobs+1)*0.5)
    n2 = Int(nobs^2)
    L = zeros(n1, n2)

    for j in 1:nobs
        for i in j:nobs
            u = zeros(n1)
            u[Int((j-1)*nobs + i-0.5*j*(j-1))] = 1
            T = zeros(nobs, nobs)
            T[i, j] = 1
            L += u*transpose(vec(T))
        end
    end
    return L
end

function commutation_matrix(n; tosparse = false)

    M = zeros(n^2, n^2)

    for i = 1:n
        for j = 1:n
            M[i + n*(j - 1), j + n*(i - 1)] = 1.0
        end
    end

    if tosparse M = sparse(M) end

    return M

end

function commutation_matrix_pre_square(A)

    n2 = size(A, 1)
    n = Int(sqrt(n2))

    ind = repeat(1:n, inner = n)
    indadd = (0:(n-1))*n
    for i in 1:n ind[((i-1)*n+1):i*n] .+= indadd end

    A_post = A[ind, :]

    return A_post

end

function commutation_matrix_pre_square_add!(B, A) # comuptes B + KₙA

    n2 = size(A, 1)
    n = Int(sqrt(n2))

    ind = repeat(1:n, inner = n)
    indadd = (0:(n-1))*n
    for i in 1:n ind[((i-1)*n+1):i*n] .+= indadd end

    @views @inbounds B .+= A[ind, :]

    return B

end

function get_commutation_lookup(n2::Int64)

    n = Int(sqrt(n2))
    ind = repeat(1:n, inner = n)
    indadd = (0:(n-1))*n
    for i in 1:n ind[((i-1)*n+1):i*n] .+= indadd end

    lookup = Dict{Int64, Int64}()

    for i in 1:n2
        j = findall(x -> (x == i), ind)[1]
        push!(lookup, i => j)
    end

    return lookup

end

function commutation_matrix_pre_square!(A::SparseMatrixCSC, lookup) # comuptes B + KₙA

    for (i, rowind) in enumerate(A.rowval)
        A.rowval[i] = lookup[rowind]
    end

end

function commutation_matrix_pre_square!(A::SparseMatrixCSC) # computes KₙA
    lookup = get_commutation_lookup(size(A, 2))
    commutation_matrix_pre_square!(A, lookup)
end

function commutation_matrix_pre_square(A::SparseMatrixCSC)
    B = copy(A)
    commutation_matrix_pre_square!(B)
    return B
end

function commutation_matrix_pre_square(A::SparseMatrixCSC, lookup)
    B = copy(A)
    commutation_matrix_pre_square!(B, lookup)
    return B
end


function commutation_matrix_pre_square_add_mt!(B, A) # comuptes B + KₙA # 0 allocations but slower

    n2 = size(A, 1)
    n = Int(sqrt(n2))

    indadd = (0:(n-1))*n

    Threads.@threads for i = 1:n
        for j = 1:n
            row = i + indadd[j]
            @views @inbounds B[row, :] .+= A[row, :]
        end
    end

    return B

end