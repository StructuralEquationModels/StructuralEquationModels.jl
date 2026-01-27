## connect to Optim.jl as backend

############################################################################################
### Types and Constructor
############################################################################################

# SemOptimizer for the Optim.jl
mutable struct SemOptimizerOptim{A, B} <: SemOptimizer{:Optim}
    algorithm::A
    options::B
end

"""
    SemOptimizer(;
        engine = :Optim,
        algorithm = LBFGS(),
        options = Optim.Options(;f_reltol = 1e-10, x_abstol = 1.5e-8),
        kwargs...)

Connects to *Optim.jl* as the optimization engine.

For more information on the available algorithms and options,
see the [*Optim.jl* docs](https://julianlsolvers.github.io/Optim.jl/stable/).

# Arguments
- `algorithm`: optimization algorithm from *Optim.jl*
- `options::Optim.Options`: options for the optimization algorithm

# Examples
```julia
# hessian based optimization with backtracking linesearch and modified initial step size
using Optim, LineSearches

my_newton_optimizer = SemOptimizer(
    engine = :Optim,
    algorithm = Newton(
        ;linesearch = BackTracking(order=3),
        alphaguess = InitialHagerZhang()
    )
)
```

# Constrained optimization
When using the `Fminbox` or `SAMIN` constrained optimization algorithms,
the vector or dictionary of lower and upper bounds for each model parameter can be specified
via `lower_bounds` and `upper_bounds` keyword arguments.
Alternatively, the `lower_bound` and `upper_bound` keyword arguments can be used to specify
the default bound for all non-variance model parameters,
and the `variance_lower_bound` and `variance_upper_bound` keyword --
for the variance parameters (the diagonal of the *S* matrix).
"""
SemOptimizerOptim(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_reltol = 1e-10, x_abstol = 1.5e-8),
    kwargs...,
) = SemOptimizerOptim(algorithm, options)

sem_optimizer_subtype(::Val{:Optim}) = SemOptimizerOptim

############################################################################################
### Recommended methods
############################################################################################

update_observed(optimizer::SemOptimizerOptim, observed::SemObserved; kwargs...) = optimizer

############################################################################################
### additional methods
############################################################################################

algorithm(optimizer::SemOptimizerOptim) = optimizer.algorithm
options(optimizer::SemOptimizerOptim) = optimizer.options

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

function fit(
    optim::SemOptimizerOptim,
    model::AbstractSem,
    start_params::AbstractVector;
    lower_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    upper_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    lower_bound = -Inf,
    upper_bound = Inf,
    variance_lower_bound::Number = -Inf,
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
