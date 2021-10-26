# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,C,L,M,M2,U,V,B} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    mult::M
    logdets::L #logdets of implied covmats
    meandiff::M2
    objective::U
    grad::V
    approx_H::B
end

############################################################################
### Constructors
############################################################################

function SemML(observed::T, objective, grad; approx_H = false) where {T <: SemObs}
    isnothing(observed.obs_mean) ?
        meandiff = nothing :
        meandiff = copy(observed.obs_mean)
    return SemML(
        copy(observed.obs_cov),
        nothing,
        copy(observed.obs_cov),
        nothing,
        meandiff,
        copy(objective),
        copy(grad),
        approx_H
        )
end

############################################################################
### functors
############################################################################

function (semml::SemML)(par, F, G, H, model, weight = nothing)
    semml.inverses .= model.imply.Σ
    a = cholesky!(Symmetric(semml.inverses); check = false)

    if !isposdef(a)
        if !isnothing(G) G .+= 0.0 end
        if !isnothing(H) stop("analytic hessian of ML is not implemented (yet)") end
        if !isnothing(F) return Inf end
    end

    ld = logdet(a)
    semml.inverses .= LinearAlgebra.inv!(a)

    if !isnothing(G) && !isnothing(H)
        J = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'
        grad = J*model.imply.∇Σ
        if !isnothing(weight)
            grad = weight*grad
        end
        G .+= grad'
        if semml.approx_H
            hessian = 2*model.imply.∇Σ'*kron(semml.inverses, semml.inverses)*model.imply.∇Σ
        end
        if !semml.approx_H
            M = semml.inverses*model.observed.obs_cov*semml.inverses
            H_outer = 
                2*kron(M, semml.inverses) - 
                kron(semml.inverses, semml.inverses)
            hessian = model.imply.∇Σ'*H_outer*model.imply.∇Σ
            model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
            hessian = hessian + model.imply.∇²Σ 
        end
        if !isnothing(weight)
            hessian = weight*hessian
        end
        H .+= hessian
    end

    if !isnothing(G) && isnothing(H)
        grad = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'*model.imply.∇Σ
        if !isnothing(weight)
            grad = weight*grad
        end
        G .+= grad'
    end

    if isnothing(G) && !isnothing(H)
        J = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'
        if semml.approx_H
            hessian = 2*model.imply.∇Σ'*kron(semml.inverses, semml.inverses)*model.imply.∇Σ
        end
        if !semml.approx_H
            M = semml.inverses*model.observed.obs_cov*semml.inverses
            H_outer = 
                2*kron(M, semml.inverses) - 
                kron(semml.inverses, semml.inverses)
            hessian = model.imply.∇Σ'*H_outer*model.imply.∇Σ
            model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
            hessian = hessian + model.imply.∇²Σ 
        end
        if !isnothing(weight)
            hessian = weight*hessian
        end
        H .+= hessian
    end

    if !isnothing(F)
        mul!(semml.mult, semml.inverses, model.observed.obs_cov)
        F = ld + tr(semml.mult)
        if !isnothing(model.imply.μ)
            @. semml.meandiff = model.observed.m - model.imply.μ
            F_mean = semml.meandiff'*semml.inverses*semml.meandiff
            F += F_mean
        end
        if !isnothing(weight)
            F = weight*F
        end
        return F
    end
end