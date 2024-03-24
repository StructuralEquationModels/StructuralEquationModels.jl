## connect to Optim.jl as backend

function SemFit(
        optimization_result::Optim.MultivariateOptimizationResults,
        model::AbstractSem,
        start_val)
    return SemFit(
        optimization_result.minimum,
        optimization_result.minimizer,
        start_val,
        model,
        optimization_result
    )
end

optimizer(res::Optim.MultivariateOptimizationResults) = Optim.summary(res)
n_iterations(res::Optim.MultivariateOptimizationResults) = Optim.iterations(res)
convergence(res::Optim.MultivariateOptimizationResults) = Optim.converged(res)

function sem_fit(
        optim::SemOptimizerOptim,
        model::AbstractSem,
        start_params::AbstractVector;
        kwargs...)

    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
                start_params,
                model.optimizer.algorithm,
                model.optimizer.options)
    return SemFit(result, model, start_params)

end
