##### weighted least squares

############################################################################
### Types
############################################################################

struct SemSNLLS{Vt, St, B, FT, GT, HT} <: SemLossFunction
    V::Vt
    s::St
    sᵀV::B

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemSNLLS(
    observed::T, 
    n_par; 
    V = nothing, 
    meanstructure = false,
    parameter_type = Float64) where {T <: SemObs}
    
    ind = findall(!iszero, LowerTriangular(observed.obs_cov))
    s = observed.obs_cov[ind]

    # compute V
    if isnothing(V)
        if !meanstructure
            D = duplication_matrix(observed.n_man)
            S = inv(observed.obs_cov)
            S = kron(S, S)
            V = 0.5*(D'*S*D)
        else
            D = duplication_matrix(observed.n_man)
            S = inv(observed.obs_cov)
            S = kron(S, S)
            V_σ = 0.5*(D'*S*D)
            V_μ = S
            a, b = size(V_σ)
            c, d = size(V_μ)
            V = [V_σ            zeros(a, d)
                zeros(c, b)     V_μ]
        end
    end

    if meanstructure
        s = vcat(s, observed.obs_mean)
    end

    sᵀV = transpose(s)*V



    return SemSNLLS(
        V, 
        s,
        sᵀV,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (semsnlls::SemSNLLS)(par, F, G, H, model)

    if !isnothing(H)
        @error "analytic hessian not implemented for SNLLS" 
    end
    
    outer = semsnlls.sᵀV*model.imply.G
    b = cholesky(Symmetric(model.imply.G'*semsnlls.V*model.imply.G))
    a = b\(transpose(outer))

    # without meanstructure

    if !isnothing(F)
        semsnlls.F[1] = -outer*a
    end

    if !isnothing(G)
        semsnlls.G .= 2*vec((semsnlls.V*model.imply.G*a - semsnlls.V*semsnlls.s)*a')'*model.imply.∇G
    end

end

############################################################################
### additional functions
############################################################################