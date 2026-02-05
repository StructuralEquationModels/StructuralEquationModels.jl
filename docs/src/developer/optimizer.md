# Custom optimizer types

The optimizer part of a model connects it to the optimization engine.
Let's say we want to implement a new optimizer as `SemOptimizerMyopt`.

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
    algorithm = ...,
    options = ...,
    kwargs...,
) = SemOptimizerMyopt(algorithm, options)

struct MyoptResult{O <: SemOptimizerMyopt} <: SEM.SemOptimizerResult{O}
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

Note that `SemOptimizerMyopt` is defined as a subtype of [`SemOptimizer{:Myopt}`](@ref SEM.SemOptimizer)`,
and `SEM.sem_optimizer_subtype(::Val{:Myopt})` returns `SemOptimizerMyopt`.
This instructs *SEM.jl* to use `SemOptimizerMyopt` when `:Myopt` is specified as the engine for
model fitting: `fit(..., engine = :Myopt)`.

A method for `update_observed` and additional methods might be usefull, but are not necessary.

Now comes the essential part: we need to provide the [`fit`](@ref) method with `SemOptimizerMyopt`
as the first positional argument.

```julia
function fit(
    optim::SemOptimizerMyopt,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    # ... prepare the Myopt optimization problem

    myopt_res = ... # fit the problem with the Myopt engine
    minimum = ... # extract the minimum from myopt_res
    minimizer = ... # extract the solution (parameter estimates)
    optim_result = MyoptResult(optim, myopt_res, ...) # store the original Myopt result and params

    return SemFit(minimum, minimizer, start_params, model, optim_result)
end
```

This method is responsible for converting the SEM into the format required by your optimization engine,
running the optimization, extracting the solution and returning the `SemFit` object, which should package:
* the minimum of the objective at the solution
* the minimizer (the vector of the SEM parameter estimates)
* the starting values
* the SEM model
* `MyoptResult` object with any relevant engine-specific details you want to preserve

In addition, you might want to provide methods to access engine-specific properties stored in `MyoptResult`:

```julia
algorithm_name(res::MyoptResult) = ...
n_iterations(res::MyoptResult) = ...
convergence(res::MyoptResult) = ...
```
