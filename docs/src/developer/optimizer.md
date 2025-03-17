# Custom optimizer types

The optimizer part of a model connects it to the optimization backend. 
Let's say we want to implement a new optimizer as `SemOptimizerName`. The first part of the implementation is very similar to loss functions, so we just show the implementation of `SemOptimizerOptim` here as a reference:

```julia
############################################################################################
### Types and Constructor
############################################################################################
mutable struct SemOptimizerName{A, B} <: SemOptimizer{:Name}
    algorithm::A
    options::B
end

SemOptimizer{:Name}(args...; kwargs...) = SemOptimizerName(args...; kwargs...)

SemOptimizerName(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_tol = 1e-10, x_tol = 1.5e-8),
    kwargs...,
) = SemOptimizerName(algorithm, options)

############################################################################################
### Recommended methods
############################################################################################

update_observed(optimizer::SemOptimizerName, observed::SemObserved; kwargs...) = optimizer

############################################################################################
### additional methods
############################################################################################

algorithm(optimizer::SemOptimizerName) = optimizer.algorithm
options(optimizer::SemOptimizerName) = optimizer.options
```

Note that your optimizer is a subtype of `SemOptimizer{:Name}`, where you can choose a `:Name` that can later be used as a keyword argument to `fit(engine = :Name)`.
Similarly, `SemOptimizer{:Name}(args...; kwargs...) = SemOptimizerName(args...; kwargs...)` should be defined as well as a constructor that uses only keyword arguments:

´´´julia
SemOptimizerName(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_tol = 1e-10, x_tol = 1.5e-8),
    kwargs...,
) = SemOptimizerName(algorithm, options)
´´´
A method for `update_observed` and additional methods might be usefull, but are not necessary.

Now comes the substantive part: We need to provide a method for `fit`:

```julia
function fit(
    optim::SemOptimizerName,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    optimization_result = ...

    ...

    return SemFit(minimum, minimizer, start_params, model, optimization_result)
end
```

The method has to return a `SemFit` object that consists of the minimum of the objective at the solution, the minimizer (aka parameter estimates), the starting values, the model and the optimization result (which may be anything you desire for your specific backend).

In addition, you might want to provide methods to access properties of your optimization result:

```julia
optimizer(res::MyOptimizationResult) = ...
n_iterations(res::MyOptimizationResult) = ...
convergence(res::MyOptimizationResult) = ...
```