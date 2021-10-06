struct ∇²SemWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}},
        St <: AbstractArray}
    V::Vt
    s::St
end

function ∇²SemWLS(semwls::A) where {A <: SemWLS}
    return ∇SemWLS(semwls.V, semwls.s)
end

function (diff::∇²SemWLS)(par, H, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    model.imply.imp_fun(model.imply.imp_cov, par)
    model.imply.gradient_fun(model.imply.∇Σ, par)
    J = (-2*(diff.s-model.imply.imp_cov)'*diff.V)'
    model.imply.hessian_fun(model.imply.∇²Σ, J, par)
    H .= 2*model.imply.∇Σ'*diff.V*model.imply.∇Σ + model.imply.∇²Σ
end

struct ∇²SemWLS_approx{Vt <: Union{AbstractArray, UniformScaling{Bool}},
        St <: AbstractArray}
    V::Vt
    s::St
end

function ∇²SemWLS_approx(semwls::A) where {A <: SemWLS}
    return ∇²SemWLS_approx(semwls.V, semwls.s)
end

function (diff::∇²SemWLS_approx)(par, H, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    model.imply.imp_fun(model.imply.imp_cov, par)
    model.imply.gradient_fun(model.imply.∇Σ, par)
    H .= 2*model.imply.∇Σ'*diff.V*model.imply.∇Σ
end