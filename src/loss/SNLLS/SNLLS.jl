##### weighted least squares

############################################################################
### Types
############################################################################

struct SemWLS{Vt, St, B, C, FT, GT, HT} <: SemLossFunction
    V::Vt
    s::St
    approx_H::B
    V_μ::C

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemWLS(observed::T, n_par; V = nothing, meanstructure = false, V_μ = nothing, approx_H = false, parameter_type = Float64) where {T <: SemObs}
    ind = CartesianIndices(observed.obs_cov)
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = observed.obs_cov[ind]

    # compute V here
    if isnothing(V)
        D = duplication_matrix(observed.n_man)
        S = inv(observed.obs_cov)
        S = kron(S, S)
        V = 0.5*(D'*S*D)
    end

    if meanstructure
        if isnothing(V_μ)
            V_μ = inv(observed.obs_cov)
        end
    else
        V_μ = nothing
    end

    return SemWLS(
        V, 
        s, 
        approx_H, 
        V_μ,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (semwls::SemWLS)(par, F, G, H, model)
    σ_diff = semwls.s - model.imply.Σ
    if isnothing(semwls.V_μ)
    # without meanstructure
        if !isnothing(G) && !isnothing(H)
            J = (-2*(σ_diff)'*semwls.V)'
            G = model.imply.∇Σ'*J
            semwls.G .= G
            H = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
            if !semwls.approx_H
                model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                H += model.imply.∇²Σ 
            end
            semwls.H .= H
        end
        if isnothing(G) && !isnothing(H)
            H = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
            if !semwls.approx_H
                J = (-2*(σ_diff)'*semwls.V)'
                model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                H += model.imply.∇²Σ
            end
            semwls.H .= H
        end
        if !isnothing(G) && isnothing(H)
            G = (-2*(σ_diff)'*semwls.V*model.imply.∇Σ)'
            semwls.G .= G
        end
        if !isnothing(F)
            F = dot(σ_diff, semwls.V, σ_diff)
            semwls.F[1] = F
        end
    else
    # with meanstructure
    μ_diff = model.observed.obs_mean - model.imply.μ
        if !isnothing(H) stop("hessian of GLS with meanstructure is not available") end
        if !isnothing(G)
            G = -2*(σ_diff'*semwls.V*model.imply.∇Σ + μ_diff'*semwls.V_μ*model.imply.∇μ)'
            semwls.G .= G
        end
        if !isnothing(F)
            F = σ_diff'*semwls.V*σ_diff + μ_diff'*semwls.V_μ*μ_diff
            semwls.F[1] = F
        end
    end
end

############## SWLS
struct SemSWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}},
        St <: AbstractArray} <: LossFunction
    V::Vt
    sᵀV::St
    #s::St
end


### Constructor
function SemSWLS(observed::T, objective, grad; V = nothing) where {T <: SemObs}
    ind = CartesianIndices(observed.obs_cov)
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = observed.obs_cov[ind]

    if isnothing(V)
        D = duplication_matrix(observed.n_man)
        S = inv(observed.obs_cov)
        S = kron(S,S)
        V = 0.5*(D'*S*D)
    end

    sᵀV = transpose(s)*V
    return SemSWLS(V, sᵀV)
end

### Loss
function (semswls::SemSWLS)(par, model)
    # s = semswls.s
    # V = semswls.V
    # G = model.imply.G
    outer = semswls.sᵀV*model.imply.G
    b = cholesky(Symmetric(model.imply.G'*semswls.V*model.imply.G))
    inner = b\outer'
    F = -outer*inner
    # F = transpose(s)*V*s - transpose(s)*V*G*inv(transpose(G)*V*G)*(transpose(G)*V*s)
    #=     θ = inv(transpose(G)*V*G)*(transpose(G)*V*s)
    σ = G*θ
    diff = s - σ
    F = dot(diff, V, diff) =#
    return F
end

struct ∇SemWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}},
    St <: AbstractArray, Lt <: AbstractArray} <: DiffFunction
    V::Vt
    s::St
    L::Lt
    end

    function ∇SemWLS(semwls::A, nobs) where {A <: SemWLS}
    L = elimination_matrix(nobs)
    return ∇SemWLS(semwls.V, semwls.s, L)
    end

    function (diff::∇SemWLS)(par, grad::AbstractVector, model::Sem{O, I, L, D}) where
        {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    model.imply.imp_fun(model.imply.imp_cov, par)
    model.imply.gradient_fun(model.imply.∇Σ, par)
    grad .= (-2*(diff.s-model.imply.imp_cov)'*diff.V*model.imply.∇Σ)'
end