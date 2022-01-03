############################################################################
### Types
############################################################################

struct SNLLS{N1, N2, N3, I1, I2, I3, M1, M2, M3, M4} <: Imply

    q_directed::N1,
    q_undirected::N2,
    size_σ::N3,

    A_indices_linear::I1,
    A_indices_cartesian::I2,
    S_indices::I3,

    A_pre::M1,
    I_A::M2,
    G::M3,
    ∇G::M4

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

    acyclic = isone(det(I-A_pre))

    # check if A is lower or upper triangular
    if iszero(A_pre[.!tril(ones(Bool,10,10))])
        A_pre = LowerTriangular(A_pre)
    elseif iszero(A_pre[.!tril(ones(Bool,10,10))'])
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits."
    end

    I_A = zeros(n_nod, n_nod)

    size_σ = Int(0.5*(n_obs^2+n_obs))
    G = zeros(size_σ, q_undirected)
    # TODO: analyze sparsity pattern of G

    if gradient
        ∇G = zeros(size_σ*q_undirected, q_directed)
        # TODO: analyze sparsity pattern of ∇G
    else
        ∇G = nothing
    end

    # μ
    if !isnothing(M)

        if check_constants(M)
            @error "constant mean parameters different from zero are not allowed in SNLLS"
        end

        M_indices = get_parameter_indices(parameters, M)
        #M_indices = [convert(Vector{Int}, indices) for indices in M_indices]
        #M_pre = zeros(size(M)...)

        M_indices

        if gradient

        else
            M_indices = nothing
            M_pre = nothing
        end

    else

    end

    return SNLLS(
        q_directed,
        q_undirected,
        size_σ,

        A_indices_linear,
        A_indices_cartesian,
        S_indices,

        A_pre,
        I_A,
        G,
        ∇G
    )
end

############################################################################
### functors
############################################################################

function (imply::SNLLS)(par, F, G, H, model)

    fill_matrix(
        imply.A,
        imply.A_indices,
        par)

    imply.I_A .= inv(I - imply.A)

    fill_G!(
        imply.G,
        imply.q_undirected,
        imply.size_σ,
        imply.S_indices,
        imply.σ_indices,
        imply.I_A)

    if !isnothing(imply.μ)

        fill_G_μ!(imply.G_μ,
        imply.q_mean, 
        imply.M_indices, 
        imply.F*imply.I_A)
        
    end

    if !isnothing(G)
        fill_∇G!(
            imply.∇G,
            imply.q_undirected,
            imply.size_σ,
            imply.q_directed,
            imply.S_indices,
            imply.σ_indices,
            imply.A_indices,
            imply.I_A)
    end

end

############################################################################
### additional functions
############################################################################

function fill_G!(G, q_undirected, size_σ, S_indices, σ_indices, I_A)

    for s in 1:q_undirected
        l, k = S_indices[s][1], S_indices[s][2]
        # rows
        for r in 1:size_σ
            i, j = σ_indices[r][1], σ_indices[r][2]
            G[r, s] = I_A[i, l]*I_A[j, k]
            if l != k
                G[r, s] += I_A[i, k]*I_A[j, l]
            end
        end
    end

end

function fill_G_μ!(G_μ, q_mean, M_indices, FI_A)

    for i in 1:q_mean
        for j in M_indices[i]
            G_μ[:, i] += FI_A[:, j]
        end
    end

end

function fill_∇G!(∇G, q_undirected, size_σ, q_directed, S_indices, σ_indices, A_indices, I_A)

    for s in 1:q_undirected
        l, k = S_indices[s][1], S_indices[s][2]

        for r in 1:size_σ
            i, j = σ_indices[r][1], σ_indices[r][2]
            t = (s-1)*size_σ + r

            for m in 1:q_directed
                u, v = A_indices[m][1], A_indices[m][2]
                ∇G[t, m] = I_A[i, u]*I_A[v, l]*I_A[j, k] + I_A[i, l]*I_A[j, u]*I_A[v, k]
                if l != k
                    ∇G[t, m] += I_A[i, u]*I_A[v, k]*I_A[j, l] + I_A[i, k]*I_A[j, u]*I_A[v, l]
                end
            end

        end

    end

end