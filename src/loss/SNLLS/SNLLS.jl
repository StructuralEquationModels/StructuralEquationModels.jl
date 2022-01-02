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
    
    ind = findall(!iszero, LowerTriangular(S))
    s = S[ind]

    # compute V
    if isnothing(V)
        D = duplication_matrix(observed.n_man)
        S = inv(observed.obs_cov)
        S = kron(S, S)
        V = 0.5*(D'*S*D)
    end

    sᵀV = transpose(s)*V

    #if meanstructure

    #else

    #end

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