## connect to Optim.jl as backend

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
    model::AbstractSem,
    start_params::AbstractVector;
    lower_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    upper_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    variance_lower_bound::Float64 = 0.0,
    lower_bound = -Inf,
    upper_bound = Inf,
    kwargs...,
)

    # setup lower/upper bounds if the algorithm supports it
    if optim.algorithm isa Optim.Fminbox || optim.algorithm isa Optim.SAMIN
        lbounds = SEM.lower_bounds(
            lower_bounds,
            model,
            default = lower_bound,
            variance_default = variance_lower_bound,
        )
        ubounds = SEM.upper_bounds(upper_bounds, model, default = upper_bound)
        result = Optim.optimize(
            Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
            lbounds,
            ubounds,
            start_params,
            model.optimizer.algorithm,
            model.optimizer.options,
        )
    else
        result = Optim.optimize(
            Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
            start_params,
            model.optimizer.algorithm,
            model.optimizer.options,
        )
    end
    return SemFit(result, model, start_params)
end
