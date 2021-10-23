# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{
        INV <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        L <: Union{AbstractArray, Nothing},
        M <: Union{AbstractArray, Nothing},
        M2 <: Union{AbstractArray, Nothing},
        U,
        V} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    mult::M
    logdets::L #logdets of implied covmats
    meandiff::M2
    objective::U
    grad::V
end

############################################################################
### Constructors
############################################################################

function SemML(observed::T, objective, grad) where {T <: SemObs}
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
    if !isnothing(G)
        grad = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'*model.imply.∇Σ
        if !isnothing(weight)
            grad = weight*grad
        end
        G .+= grad'
    end
    if !isnothing(H) stop("analytic hessian of ML is not implemented (yet)") end
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