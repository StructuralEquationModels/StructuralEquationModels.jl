## connect do Optim.jl as backend

function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemForwardDiff}
    result = Optim.optimize(
                par -> model(par),
                model.imply.start_val,
                model.diff.algorithm,
                autodiff = :forward,
                model.diff.options)
    return result
end

function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}
    result = Optim.optimize(
                par -> model(par),
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)
    return result
end

function sem_fit(model::A, g!) where
    {A <: AbstractSem}
    result = Optim.optimize(
                par -> model(par),
                g!,
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)#;
                #inplace = false)
    return result
end

function sem_fit(model::A, start_val::B) where
    {A <: AbstractSem, B <: AbstractArray}
    result = Optim.optimize(
                par -> model(par),
                start_val,
                model.sem_vec[1].diff.algorithm,
                model.sem_vec[1].diff.options)
    return result
end

function sem_fit(model::A, g!, h!) where
    {A <: AbstractSem}
    result = Optim.optimize(
                par -> model(par),
                g!,
                h!,
                model.imply.start_val,
                Newton(),
                model.diff.options)#;
                #inplace = false)
    return result
end

#function sem_fit(model::Sem{O, I, L, D}) where
#    {O <: SemObs, L <: Loss, I <: Imply, D <: SemReverseDiff}
#    result = optimize(
#                par -> model(par),
#                par -> Zygote.gradient(model, par)[1],
#                model.imply.start_val,
#                model.diff.algorithm,
#                model.diff.options;
#                inplace = false)
#    return result
#end

function sem_fit(model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    result = Optim.optimize(
                Optim.only_fg!(model),
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)
    return result
end

#function sem_fit(model::A, start_val) where
#    {A <: AbstractSem}
#    result = optimize(
#                par -> model(par),
#                start_val,
#                model.sem_vec[1].diff.algorithm,
#                model.sem_vec[1].diff.options)
#    return result
#end
