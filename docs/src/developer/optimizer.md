# Custom optimizer types

The optimizer part of a model connects it to the optimization backend.
Let's say we want to implement a new optimizer as `SemOptimizerMyopt`.
The first part of the implementation is very similar to loss functions,
so we just show the implementation of `SemOptimizerOptim` here as a reference:

```julia
############################################################################################
### Types and Constructor
############################################################################################
struct SemOptimizerMyopt{A, B} <: SemOptimizer{:Myopt}
    algorithm::A
    options::B
end

SEM.sem_optimizer_subtype(::Val{:Myopt}) = SemOptimizerMyopt

SemOptimizerMyopt(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_reltol = 1e-10, x_abstol = 1.5e-8),
    kwargs...,
) = SemOptimizerMyopt(algorithm, options)

struct MyOptResult{O <: SemOptimizerMyopt} <: SEM.SemOptimizerResult{O}
    optimizer::O
    ...
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(optimizer::SemOptimizerMyopt, observed::SemObserved; kwargs...) = optimizer

############################################################################################
### additional methods
############################################################################################

options(optimizer::SemOptimizerMyopt) = optimizer.options
```

Note that your optimizer is a subtype of `SemOptimizer{:Myopt}`,
where you can choose a `:Myopt` that can later be used as a keyword argument to `fit(engine = :Myopt)`.
Similarly, `SemOptimizer{:Myopt}(args...; kwargs...) = SemOptimizerMyopt(args...; kwargs...)`
should be defined as well as a constructor that uses only keyword arguments:

```julia
SemOptimizerMyopt(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_reltol = 1e-10, x_abstol = 1.5e-8),
    kwargs...,
) = SemOptimizerMyopt(algorithm, options)
```
A method for `update_observed` and additional methods might be usefull, but are not necessary.

Now comes the substantive part: We need to provide a method for `fit`:

```julia
function fit(
    optim::SemOptimizerMyopt,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    ...

    optimization_result = MyoptResult(optim, ...)

    return SemFit(minimum, minimizer, start_params, model, optimization_result)
end
```

The method has to return a `SemFit` object that consists of the minimum of the objective at the solution, the minimizer (aka parameter estimates), the starting values, the model and the optimization result (which may be anything you desire for your specific backend).

In addition, you might want to provide methods to access properties of your optimization result:

```julia
algorithm_name(res::MyOptResult) = ...
n_iterations(res::MyOptResult) = ...
convergence(res::MyOptResult) = ...
```