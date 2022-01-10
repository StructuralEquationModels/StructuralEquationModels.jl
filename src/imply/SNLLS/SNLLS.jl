############################################################################
### Types
############################################################################

struct SNLLS{N1, N2, N3, I1, I2, I3, I6, M1, M2, M3, M4, N4, I4, M5, I5, V1} <: SemImply

    q_directed::N1
    q_undirected::N2
    size_σ::N3

    A_indices_linear::I1
    A_indices_cartesian::I2
    S_indices::I3
    σ_indices::I6

    A_pre::M1
    I_A::M2
    G::M3
    ∇G::M4

    q_mean::N4
    M_indices::I4
    G_μ::M5
    G_μ_indices::I5

    start_val::V1

end

############################################################################
### Constructors
############################################################################

function SNLLS(
    A::Spa1,
    S::Spa2,
    F::Spa3,
    parameters,
    start_val;

    M::Spa4 = nothing,
    gradient = true

        ) where {
        Spa1 <: SparseMatrixCSC,
        Spa2 <: SparseMatrixCSC,
        Spa3 <: SparseMatrixCSC,
        Spa4 <: Union{Nothing, AbstractArray}
        }

    n_var, n_nod = size(F)

    A = Matrix(A)
    S = Matrix(S)
    F = Matrix(F); F = convert(Matrix{Float64}, F)

    if check_constants(S)
        @error "constant variance/covariance parameters different from zero are not allowed in SNLLS"
    end

    σ_indices = findall(isone, LowerTriangular(ones(n_var, n_var)))

    # store the indices of the parameters
    A_indices = get_parameter_indices(parameters, A)
    S_indices = get_parameter_indices(parameters, S; index_function = eachindex_lower, linear_indices = true)

    A_indices = [convert(Vector{Int}, indices) for indices in A_indices]
    S_indices = [convert(Vector{Int}, indices) for indices in S_indices]

    A_pars, S_pars = get_partition(A_indices, S_indices)

    # dimension of undirected and directed parameters
    q = size(start_val)
    q_undirected = length(S_pars)
    q_directed = length(A_pars)

    A_indices_linear = A_indices[A_pars]
    A_indices_cartesian = linear2cartesian.(A_indices_linear, [A])

    S_indices = linear2cartesian.(S_indices, [S])
    S_indices = S_indices[S_pars]

    # A matrix
    A_pre = zeros(size(A)...)
    set_constants!(A, A_pre)

    A_rand = copy(A_pre)
    randpar = rand(length(start_val))

    fill_matrix(
        A_rand,
        A_indices_linear,
        randpar)

    acyclic = isone(det(I-A_rand))

    # check if A is lower or upper triangular
    if iszero(A_rand[.!tril(ones(Bool, size(A)...))])
        A_pre = LowerTriangular(A_pre)
    elseif iszero(A_rand[.!tril(ones(Bool, size(A)...))'])
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits."
    end

    I_A = zeros(n_nod, n_nod)

    size_σ = Int(0.5*(n_var^2+n_var))

    if !isnothing(M)

        if check_constants(M)
            @error "constant mean parameters different from zero are not allowed in SNLLS"
        end
    
        M_indices = get_parameter_indices(parameters, M)
        M_indices = [convert(Vector{Int}, indices) for indices in M_indices]
        M_pars = get_partition(M_indices)
        M_indices = M_indices[M_pars]
        q_mean = length(M_pars)
    
        G = zeros(size_σ+n_var, q_undirected+q_mean)
    
        G_μ = zeros(n_var, q_mean)
    
        G_μ_indices = CartesianIndices((size_σ .+ (1:n_var), q_undirected .+ (1:q_mean)))
        # TODO: analyze sparsity pattern of G
    
        if gradient
            ∇G = zeros((size_σ+n_var)*(q_undirected+q_mean), q_directed)
            # TODO: analyze sparsity pattern of ∇G
        else
            ∇G = nothing
        end
    
    else
        
        q_mean = nothing
        M_indices = nothing
        G_μ = nothing
        G_μ_indices = nothing
    
        G = zeros(size_σ, q_undirected)
        # TODO: analyze sparsity pattern of G
    
        if gradient
            ∇G = zeros(size_σ*q_undirected, q_directed)
            # TODO: analyze sparsity pattern of ∇G
        else
            ∇G = nothing
        end
    
    end

    return SNLLS(
        q_directed,
        q_undirected,
        size_σ,

        A_indices_linear,
        A_indices_cartesian,
        S_indices,
        σ_indices,

        A_pre,
        I_A,
        G,
        ∇G,

        q_mean,
        M_indices,
        G_μ,
        G_μ_indices,

        start_val
    )
end

############################################################################
### functors
############################################################################

function (imply::SNLLS)(par, F, G, H, model)

    fill_matrix(
        imply.A_pre,
        imply.A_indices_linear,
        par)

    imply.I_A .= inv(I - imply.A_pre)

    fill_G!(
        imply.G,
        imply.q_undirected,
        imply.size_σ,
        imply.S_indices,
        imply.σ_indices,
        imply.I_A)

    if !isnothing(imply.M_indices)

        FI_A = imply.I_A[1:Int(model.observed.n_man), :]

        fill_G_μ!(
            imply.G_μ,
            imply.q_mean, 
            imply.M_indices, 
            FI_A)

        copyto!(imply.G, imply.G_μ_indices, imply.G_μ, CartesianIndices(imply.G_μ))

    end

    if !isnothing(G)
        fill_∇G!(
            imply.∇G,
            imply.q_undirected,
            imply.size_σ,
            imply.q_directed,
            imply.S_indices,
            imply.σ_indices,
            imply.A_indices_cartesian,
            imply.I_A,
            imply.q_mean,
            imply.M_indices,
            model.observed.n_man)
    end

end

############################################################################
### additional functions
############################################################################

function fill_G!(G, q_undirected, size_σ, S_indices, σ_indices, I_A)

    fill!(G, zero(eltype(G)))

    for s in 1:q_undirected
        for ind in S_indices[s]
            l, k = ind[1], ind[2]
            # rows
            for r in 1:size_σ
                i, j = σ_indices[r][1], σ_indices[r][2]
                G[r, s] += I_A[i, l]*I_A[j, k]
                if l != k
                    G[r, s] += I_A[i, k]*I_A[j, l]
                end
            end
        end
    end

end

function fill_G_μ!(G_μ, q_mean, M_indices, FI_A)

    fill!(G_μ, zero(eltype(G_μ)))

    for i in 1:q_mean
        for j in M_indices[i]
            G_μ[:, i] .+= FI_A[:, j]
        end
    end

end

function fill_∇G!(∇G, q_undirected, size_σ, q_directed, S_indices, σ_indices, A_indices, I_A, q_mean::Nothing, M_indices::Nothing, n_var)

    fill!(∇G, zero(eltype(∇G)))

    for s in 1:q_undirected

        for c ∈ S_indices[s]
            l, k = c[1], c[2]

            for r in 1:size_σ
                i, j = σ_indices[r][1], σ_indices[r][2]
                t = (s-1)*size_σ + r

                for m in 1:q_directed
                    u, v = A_indices[m][1][1], A_indices[m][1][2]
                    ∇G[t, m] += I_A[i, u]*I_A[v, l]*I_A[j, k] + I_A[i, l]*I_A[j, u]*I_A[v, k]
                    if l != k
                        ∇G[t, m] += I_A[i, u]*I_A[v, k]*I_A[j, l] + I_A[i, k]*I_A[j, u]*I_A[v, l]
                    end
                end

            end

        end

    end

end

function fill_∇G!(∇G, q_undirected, size_σ, q_directed, S_indices, σ_indices, A_indices, I_A, q_mean, M_indices, n_var)

    fill!(∇G, zero(eltype(∇G)))

    size_G_rows = size_σ + Int(n_var)

    for s in 1:q_undirected
        
        for c ∈ S_indices[s]
            l, k = c[1], c[2]

            for r in 1:size_σ
                i, j = σ_indices[r][1], σ_indices[r][2]
                t = (s-1)*size_G_rows + r

                for m in 1:q_directed
                    u, v = A_indices[m][1][1], A_indices[m][1][2]
                    ∇G[t, m] += I_A[i, u]*I_A[v, l]*I_A[j, k] + I_A[i, l]*I_A[j, u]*I_A[v, k]
                    if l != k
                        ∇G[t, m] += I_A[i, u]*I_A[v, k]*I_A[j, l] + I_A[i, k]*I_A[j, u]*I_A[v, l]
                    end
                end
            end
        end
    end

    for s in 1:q_mean
        
        for k ∈ M_indices[s]

            for r in 1:Int(n_var)
                t =  (q_undirected + s - 1)*size_G_rows + size_σ + r

                for m in 1:q_directed
                    u, v = A_indices[m][1][1], A_indices[m][1][2]
                    ∇G[t, m] += I_A[r, u]*I_A[v, k]
                end
            end
        end
    end

end