function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemForwardDiff}
    result = optimize(
                par -> model(par),
                model.imply.start_val,
                model.diff.algorithm,
                autodiff = :forward,
                model.diff.options)
    return result
end

function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}
    result = optimize(
                par -> model(par),
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)
    return result
end

function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemReverseDiff}
    result = optimize(
                par -> model(par),
                par -> Zygote.gradient(model, par)[1],
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options;
                inplace = false)
    return result
end
