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
    lower_bound = -Inf,
    upper_bound = Inf,
    variance_lower_bound::Number = 0.0,
    variance_upper_bound::Number = Inf,
    kwargs...,
)
    # setup lower/upper bounds if the algorithm supports it
    if optim.algorithm isa Optim.Fminbox || optim.algorithm isa Optim.SAMIN
        lbounds = prepare_param_bounds(
            Val(:lower),
            lower_bounds,
            model,
            default = lower_bound,
            variance_default = variance_lower_bound,
        )
        ubounds = prepare_param_bounds(
            Val(:upper),
            upper_bounds,
            model,
            default = upper_bound,
            variance_default = variance_upper_bound,
        )
        start_params = clamp.(start_params, lbounds, ubounds)
        result = Optim.optimize(
            Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
            lbounds,
            ubounds,
            start_params,
            optim.algorithm,
            optim.options,
        )
    else
        result = Optim.optimize(
            Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
            start_params,
            optim.algorithm,
            optim.options,
        )
    end
    return SemFit(result, model, start_params)
end
