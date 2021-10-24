##### weighted least squares

############################################################################
### Types
############################################################################

struct SemWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}}, St <: AbstractArray, B} <: SemLossFunction
    V::Vt
    s::St
    approx_H::B
end

############################################################################
### Constructors
############################################################################

function SemWLS(observed::T; V = nothing, approx_H = false) where {T <: SemObs}
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
    return SemWLS(V, s, approx_H)
end

############################################################################
### functors
############################################################################

function (semwls::SemWLS)(par, F, G, H, model, weight = nothing)
    σ_diff = semwls.s - model.imply.Σ
    if !isnothing(G) && !isnothing(H)
        J = (-2*(σ_diff)'*semwls.V)'
        grad = model.imply.∇Σ'*J
        if !isnothing(weight) grad = weight*grad end
        G .+= grad
        hessian = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
        if !approx_H hessian += model.imply.∇²Σ end
        if !isnothing(weight) hessian = weight*hessian end
        H .+= hessian
    end
    if  isnothing(G) && !isnothing(H)
        J = (-2*(σ_diff)'*semwls.V)'
        hessian = 2*model.imply.∇Σ'*semwls.V*model.imply.∇Σ
        if !approx_H hessian += model.imply.∇²Σ end
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
end