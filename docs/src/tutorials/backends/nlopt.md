# Using NLopt.jl

[`SemOptimizerNLopt`](@ref) implements the connection to `NLopt.jl`.
It takes a bunch of arguments:

```julia
    •  algorithm: optimization algorithm

    •  options::Dict{Symbol, Any}: options for the optimization algorithm

    •  local_algorithm: local optimization algorithm

    •  local_options::Dict{Symbol, Any}: options for the local optimization algorithm

    •  equality_constraints::Vector{NLoptConstraint}: vector of equality constraints

    •  inequality_constraints::Vector{NLoptConstraint}: vector of inequality constraints
```
Constraints are explained in the section on [Constrained optimization](@ref).

The defaults are LBFGS as the optimization algorithm and the standard options from `NLopt.jl`.
We can choose something different:

```julia
my_optimizer = SemOptimizerNLopt(;
    algorithm = :AUGLAG,
    options = Dict(:maxeval => 200),
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6)
)
```

This uses an augmented lagrangian method with LBFGS as the local optimization algorithm, stops at a maximum of 200 evaluations and uses a relative tolerance of the objective value of `1e-6` as the stopping criterion for the local algorithm.

In the NLopt docs, you can find explanations about the different [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) and a [tutorial](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/) that also explains the different options.

To choose an algorithm, just pass its name without the 'NLOPT_' prefix (for example, 'NLOPT_LD_SLSQP' can be used by passing `algorithm = :LD_SLSQP`).

The README of the [julia package](https://github.com/JuliaOpt/NLopt.jl) may also be helpful, and provides a list of options:

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

 # Constrained optimization