## connect do Optim.jl as backend
function sem_wrap_nlopt(par, G, sem::AbstractSem)
    need_gradient = length(G) != 0
    sem(par, true, need_gradient, false)
    if need_gradient G .= gradient(sem) end
    return objective(sem)
end

function SemFit_NLopt(optimization_result, model::AbstractSem)
    return SemFit(
        optimization_result[1],
        optimization_result[2],
        model,
        optimization_result
    )
end

function sem_fit(
            model::Sem{O, I, L, D}; 
            ftol_rel = 1e-10,
            xtol_rel = 1.5e-8,
            lower = nothing,
            upper = nothing,
            local_algo = nothing,
            maxeval = 200) where {O, I, L, D <: SemDiffNLopt}

    opt = NLopt.Opt(model.diff.algorithm, length(model.imply.start_val))
    #cache = FiniteDiff.GradientCache(start)
    opt.min_objective = (par, G) -> sem_wrap_nlopt(par, G, model)
    opt.ftol_rel = ftol_rel
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval
    !isnothing(lower) ? opt.lower_bounds = lower : nothing
    !isnothing(upper) ? opt.upper_bounds = upper : nothing
    !isnothing(local_algo) ? opt.local_optimizer = NLopt.Opt(local_algo, length(model.imply.start_val)) : nothing
    result = NLopt.optimize(opt, model.imply.start_val)
    return SemFit_NLopt(result, model)
end