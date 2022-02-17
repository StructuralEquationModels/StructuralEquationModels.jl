## connect do Optim.jl as backend
function sem_wrap_optim(par, F, G, H, sem::AbstractSem)
    sem(par, F, G, H)
    if !isnothing(G) G .= gradient(sem) end
    if !isnothing(H) H .= hessian(sem) end
    if !isnothing(F) return objective(sem) end
end

function SemFit(optimization_result::Optim.MultivariateOptimizationResults, model::AbstractSem)
    return SemFit(
        optimization_result.minimum,
        optimization_result.minimizer,
        model,
        optimization_result
    )
end

function sem_fit(model::Sem{O, I, L, D}) where {O, I, L, D <: SemDiffOptim}
    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model)
end

function sem_fit(model::SemFiniteDiff{O, I, L, D}) where {O, I, L, D <: SemDiffOptim}
    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                model.imply.start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model)
end

function sem_fit(model::SemEnsemble{N, T , V, D, S}) where {N, T, V, D <: SemDiffOptim, S}
    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                model.start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model)
end