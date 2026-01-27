############################################################################################
### Types
############################################################################################

const NLoptConstraint = Pair{Any, Number}

struct SemOptimizerNLopt <: SemOptimizer{:NLopt}
    algorithm::Symbol
    local_algorithm::Union{Symbol, Nothing}
    options::Dict{Symbol, Any}
    local_options::Dict{Symbol, Any}
    equality_constraints::Vector{NLoptConstraint}
    inequality_constraints::Vector{NLoptConstraint}
end

############################################################################################
### Constructor
############################################################################################

"""
Uses *NLopt.jl* as the optimization engine.
Only available if *NLopt.jl* is loaded in the current Julia session!

# Constructor

    SemOptimizer(;
        engine = :NLopt,
        algorithm = :LD_LBFGS,
        options = Dict{Symbol, Any}(),
        local_algorithm = nothing,
        local_options = Dict{Symbol, Any}(),
        equality_constraints = nothing,
        inequality_constraints = nothing,
        constraint_tol::Number = 0.0,
        kwargs...)

# Arguments
- `algorithm`: optimization algorithm.
- `options::Dict{Symbol, Any}`: options for the optimization algorithm
- `local_algorithm`: local optimization algorithm
- `local_options::Dict{Symbol, Any}`: options for the local optimization algorithm
- `equality_constraints: optional equality constraints
- `inequality_constraints:: optional inequality constraints
- `constraint_tol::Number`: default tolerance for constraints

## Constraints specification

Equality and inequality constraints arguments could be a single constraint or any
iterable constraints container (e.g. vector or tuple).
Each constraint could be a function or any other callable object that
takes the two input arguments:
  - the vector of the model parameters;
  - the array for the in-place calculation of the constraint gradient.
To override the default tolerance, the constraint could be specified
as a pair of the function and its tolerance: `constraint_func => tol`.

# Example
```julia
my_optimizer = SemOptimizer(engine = :NLopt)

# constrained optimization with augmented lagrangian
my_constrained_optimizer = SemOptimizer(;
    engine = :NLopt,
    algorithm = :AUGLAG,
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6),
    inequality_constraints = (my_constraint => tol),
)
```

# Usage
All algorithms and options from the *NLopt* library are available, for more information see
the [*NLopt.jl*](https://github.com/JuliaOpt/NLopt.jl) package and the
[NLopt docs](https://nlopt.readthedocs.io/en/latest/).
For information on how to use inequality and equality constraints,
see [Constrained optimization](@ref) in our online documentation.

# Extended help

## Interfaces
- `algorithm(::SemOptimizerNLopt)`
- `local_algorithm(::SemOptimizerNLopt)`
- `options(::SemOptimizerNLopt)`
- `local_options(::SemOptimizerNLopt)`
- `equality_constraints(::SemOptimizerNLopt)`
- `inequality_constraints(::SemOptimizerNLopt)`

## Implementation

Subtype of `SemOptimizer`.
"""
function SemOptimizerNLopt(;
    algorithm = :LD_LBFGS,
    local_algorithm = nothing,
    options = Dict{Symbol, Any}(),
    local_options = Dict{Symbol, Any}(),
    equality_constraints = nothing,
    inequality_constraints = nothing,
    constraint_tol::Number = 0.0,
    kwargs..., # FIXME remove the sink for unused kwargs
)
    constraint(f::Any) = f => constraint_tol
    constraint(f_and_tol::Pair) = f_and_tol

    constraints(::Nothing) = Vector{NLoptConstraint}()
    constraints(constraints) =
        applicable(iterate, constraints) && !isa(constraints, Pair) ?
        [constraint(constr) for constr in constraints] : [constraint(constraints)]

    return SemOptimizerNLopt(
        algorithm,
        local_algorithm,
        options,
        local_options,
        constraints(equality_constraints),
        constraints(inequality_constraints),
    )
end

"""
    SemOptimizer(args...; engine = :NLopt, kwargs...)

Creates SEM optimizer using [*NLopt.jl*](https://github.com/JuliaOpt/NLopt.jl).

# Extended help

See [`SemOptimizerNLopt`](@ref) for a full reference.
"""
SEM.SemOptimizer(::Val{:NLopt}, args...; kwargs...) = SemOptimizerNLopt(args...; kwargs...)

############################################################################################
### Recommended methods
############################################################################################

SEM.update_observed(optimizer::SemOptimizerNLopt, observed::SemObserved; kwargs...) =
    optimizer

############################################################################################
### additional methods
############################################################################################

SEM.algorithm(optimizer::SemOptimizerNLopt) = optimizer.algorithm
local_algorithm(optimizer::SemOptimizerNLopt) = optimizer.local_algorithm
SEM.options(optimizer::SemOptimizerNLopt) = optimizer.options
local_options(optimizer::SemOptimizerNLopt) = optimizer.local_options
equality_constraints(optimizer::SemOptimizerNLopt) = optimizer.equality_constraints
inequality_constraints(optimizer::SemOptimizerNLopt) = optimizer.inequality_constraints

struct NLoptResult
    result::Any
    problem::Any
end

SEM.optimizer(res::NLoptResult) = res.problem.algorithm
SEM.n_iterations(res::NLoptResult) = res.problem.numevals
SEM.convergence(res::NLoptResult) = res.result[3]

# construct SemFit from fitted NLopt object
function SemFit_NLopt(optimization_result, model::AbstractSem, start_val, opt)
    return SemFit(
        optimization_result[1],
        optimization_result[2],
        start_val,
        model,
        NLoptResult(optimization_result, opt),
    )
end

# fit method
function SEM.fit(
    optim::SemOptimizerNLopt,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    opt = construct_NLopt(optim.algorithm, optim.options, nparams(model))
    opt.min_objective =
        (par, G) -> SEM.evaluate!(
            zero(eltype(par)),
            !isnothing(G) && !isempty(G) ? G : nothing,
            nothing,
            model,
            par,
        )
    for (f, tol) in optim.inequality_constraints
        inequality_constraint!(opt, f, tol)
    end
    for (f, tol) in optim.equality_constraints
        equality_constraint!(opt, f, tol)
    end

    if !isnothing(optim.local_algorithm)
        opt_local =
            construct_NLopt(optim.local_algorithm, optim.local_options, nparams(model))
        opt.local_optimizer = opt_local
    end

    # fit
    result = NLopt.optimize(opt, start_params)

    return SemFit_NLopt(result, model, start_params, opt)
end

############################################################################################
### additional functions
############################################################################################

function construct_NLopt(algorithm, options, npar)
    opt = Opt(algorithm, npar)

    for (key, val) in pairs(options)
        setproperty!(opt, key, val)
    end

    return opt
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::NLoptResult)
    print(io, "Optimizer status: $(result.result[3]) \n")
    print(io, "Minimum:          $(round(result.result[1]; digits = 2)) \n")
    print(io, "Algorithm:        $(result.problem.algorithm) \n")
    print(io, "No. evaluations:  $(result.problem.numevals) \n")
end
