## connect to NLopt as backend

function (model::Sem{A, B, C, D} where {A, B, C, D <: SemFiniteDiff})(par::Vector, grad::Vector)
    if length(grad) > 0
        FiniteDiff.finite_difference_gradient!(grad, model, par)
    end
    return model(par)
end

function grad_nlopt(model, par, grad)
    if length(grad) > 0
        model(par, grad)
    end
    return model(par)
end

function sem_fit_nlopt(model::Sem{O, I, L, D}; ftol_rel = 1e-10, xtol_rel = 1.5e-8) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemDiff}

    opt = NLopt.Opt(model.diff.algorithm, length(model.imply.start_val))
    #cache = FiniteDiff.GradientCache(start)
    opt.min_objective = (x,y) -> grad_nlopt(model, x, y)
    opt.ftol_rel = ftol_rel
    opt.xtol_rel = xtol_rel
    result = NLopt.optimize(opt, model.imply.start_val)
    
    return result
end