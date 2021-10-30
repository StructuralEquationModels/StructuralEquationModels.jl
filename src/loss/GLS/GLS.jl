##### weighted least squares

############################################################################
### Types
############################################################################

struct SemWLS{Vt, St, B, C} <: SemLossFunction
    V::Vt
    s::St
    approx_H::B
    V_μ::C
end

############################################################################
### Constructors
############################################################################

function SemWLS(observed::T; V = nothing, meanstructure = false, V_μ = nothing, approx_H = false) where {T <: SemObs}
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
    return SemWLS(V, s, approx_H, V_μ)
end

############################################################################
### functors
############################################################################

function (semwls::SemWLS)(par, F, G, H, model, weight = nothing)
    σ_diff = semwls.s - model.imply.Σ
    if isnothing(semwls.V_μ)
    # without meanstructure
        if !isnothing(G) && !isnothing(H)
            J = (-2*(σ_diff)'*semwls.V)'
            grad = model.imply.∇Σ'*J
            if !isnothing(weight) grad = weight*grad end
            G .+= grad
            hessian = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
            if !semwls.approx_H
                model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                hessian += model.imply.∇²Σ 
            end
            if !isnothing(weight) hessian = weight*hessian end
            H .+= hessian
        end
        if  isnothing(G) && !isnothing(H)
            hessian = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
            if !semwls.approx_H
                J = (-2*(σ_diff)'*semwls.V)'
                model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                hessian += model.imply.∇²Σ
            end
            if !isnothing(weight) hessian = weight*hessian end
            H .+= hessian
        end
        if  !isnothing(G) && isnothing(H)
            grad = (-2*(σ_diff)'*semwls.V*model.imply.∇Σ)'
            if !isnothing(weight) grad = weight*grad end
            G .+= grad
        end
        if !isnothing(F)
            F = dot(σ_diff, semwls.V, σ_diff)
            if !isnothing(weight) F = weight*F end
            return F
        end
    else
    # with meanstructure
    μ_diff = model.observed.obs_mean - model.imply.μ
        if !isnothing(H) stop("hessian of GLS is not implemented (yet)") end
        if !isnothing(G)
            grad = -2*(σ_diff'*semwls.V*model.imply.∇Σ + μ_diff'*semwls.V_μ*model.imply.∇μ)'
            if !isnothing(weight) grad = weight*grad end
            G .+= grad
        end
        if !isnothing(F)
            F = σ_diff'*semwls.V*σ_diff + μ_diff'*semwls.V_μ*μ_diff
            if !isnothing(weight) F = weight*F end
            return F
        end
    end
end