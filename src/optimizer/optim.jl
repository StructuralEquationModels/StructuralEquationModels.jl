## connect do Optim.jl as backend
function sem_wrap_optim(par, F, G, H, sem::AbstractSem)
    sem(par, !isnothing(F), !isnothing(G), !isnothing(H))
    if !isnothing(G) G .= gradient(sem) end
    if !isnothing(H) H .= hessian(sem) end
    if !isnothing(F) return objective(sem) end
end

function SemFit(optimization_result::Optim.MultivariateOptimizationResults, model::AbstractSem, start_val)
    return SemFit(
        optimization_result.minimum,
        optimization_result.minimizer,
        start_val,
        model,
        optimization_result
    )
end

function sem_fit(model::Sem{O, I, L, D}; start_val = start_val, kwargs...) where {O, I, L, D <: SemDiffOptim}
    
    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model, start_val)

end

function sem_fit(model::SemFiniteDiff{O, I, L, D}; start_val = start_val, kwargs...) where {O, I, L, D <: SemDiffOptim}

    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model, start_val)

end

function sem_fit(model::SemEnsemble{N, T , V, D, S}; start_val = start_val, kwargs...) where {N, T, V, D <: SemDiffOptim, S}

    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                start_val,
                model.diff.algorithm,
                model.diff.options)
    return SemFit(result, model, start_val)

end