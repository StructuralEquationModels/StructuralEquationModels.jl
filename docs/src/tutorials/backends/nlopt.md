# Using NLopt.jl

When [`NLopt.jl`](https://github.com/jump-dev/NLopt.jl) is loaded in the running Julia session,
it could be used by the [`SemOptimizer`](@ref) by specifying `engine = :NLopt`
(see [NLopt-specific options](@ref SEMNLOptExt.SemOptimizerNLopt)).
Among other things, `NLopt` enables constrained optimization of the SEM models, which is
explained in the [Constrained optimization](@ref) section.

We can override the default *NLopt* algorithm (LFBGS) and instead use
the *augmented lagrangian* method with LBFGS as the *local* optimization algorithm,
stop at a maximum of 200 evaluations and use a relative tolerance of
the objective value of `1e-6` as the stopping criterion for the local algorithm:

```julia
using NLopt

my_optimizer = SemOptimizer(;
    engine = :NLopt,
    algorithm = :AUGLAG,
    options = Dict(:maxeval => 200),
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6)
)
```

To see how to use the optimizer to actually fit a model now, check out the [Model fitting](@ref) section.

In the *NLopt* docs, you can find details about the [optimization algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/),
and the [tutorial](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/) that demonstrates how to tweak their behavior.

To choose an algorithm, just pass its name without the 'NLOPT\_' prefix (for example, 'NLOPT\_LD\_SLSQP' can be used by passing `algorithm = :LD_SLSQP`).

The README of the [*NLopt.jl*](https://github.com/JuliaOpt/NLopt.jl) may also be helpful, and provides a list of options:

 - `algorithm`
 - `stopval`
 - `ftol_rel`
 - `ftol_abs`
 - `xtol_rel`
 - `xtol_abs`
 - `constrtol_abs`
 - `maxeval`
 - `maxtime`
 - `initial_step`
 - `population`
 - `seed`
 - `vector_storage`