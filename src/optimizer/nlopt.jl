## connect to NLopt as backend

function (model::Sem{A, B, C, D} where {A, B, C, D <: SemFiniteDiff})(par::Vector, grad::Vector)
    if length(grad) > 0
        FiniteDiff.finite_difference_gradient!(grad, model, par)
    end
    return model(par)
end

function sem_fit_nlopt(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemDiff}

    opt = NLopt.Opt(model.diff.algorithm, length(model.imply.start_val))
    #cache = FiniteDiff.GradientCache(start)
    opt.min_objective = (x,y) -> model(x,y)
    #opt.ftol_rel = f_tol
    result = NLopt.optimize(opt, model.imply.start_val)
    
    return result
end