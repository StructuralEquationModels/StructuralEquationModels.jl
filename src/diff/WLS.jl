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

function (diff::∇SemWLS)(par, grad, model::Sem{O, I, L, D}) where
        {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    model.imply.imp_fun(model.imply.imp_cov, par)
    model.imply.gradient_fun(model.imply.∇Σ, par)
    grad .= (-2*(diff.s-model.imply.imp_cov)'*diff.V*diff.L*model.imply.∇Σ)'
end