struct ∇SemML_2 <: DiffFunction end

function (diff::∇SemML_2)(par, grad, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    model.imply.imp_fun(model.imply.imp_cov, par)
    model.imply.gradient_fun(model.imply.∇Σ, par)
    a = cholesky(Symmetric(model.imply.imp_cov); check = false)
    if !isposdef(a)
        grad .= 0.0
    else
        Σ_inv = inv(a)
        grad .= ((vec(Σ_inv')'-vec((Σ_inv*model.observed.obs_cov*Σ_inv)')')*model.imply.∇Σ)'
    end
end