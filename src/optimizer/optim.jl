## connect to Optim.jl as backend
function sem_wrap_optim(par, F, G, H, model::AbstractSem)
    if !isnothing(F)
        if !isnothing(G)
            if !isnothing(H)
                return objective_gradient_hessian!(G, H, model, par)
            else
                return objective_gradient!(G, model, par)
            end
        else
            if !isnothing(H)
                return objective_hessian!(H, model, par)
            else
                return objective!(model, par)
            end
        end
    else
        if !isnothing(G)
            if !isnothing(H)
                gradient_hessian!(G, H, model, par)
            else
                gradient!(G, model, par)
            end
        end
    end
    return nothing
end

function SemFit(
    optimization_result::Optim.MultivariateOptimizationResults,
    model::AbstractSem,
    start_val,
)
    return SemFit(
        optimization_result.minimum,
        optimization_result.minimizer,
        start_val,
        model,
        optimization_result,
    )
end

optimizer(res::Optim.MultivariateOptimizationResults) = Optim.summary(res)
n_iterations(res::Optim.MultivariateOptimizationResults) = Optim.iterations(res)
convergence(res::Optim.MultivariateOptimizationResults) = Optim.converged(res)

function sem_fit(
    optim::SemOptimizerOptim,
    model::AbstractSem;
    start_val = start_val,
    kwargs...,
)
    if !isa(start_val, AbstractVector)
        start_val = start_val(model; kwargs...)
    end

    result = Optim.optimize(
        Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
        start_val,
        model.optimizer.algorithm,
        model.optimizer.options,
    )
    return SemFit(result, model, start_val)
end
