"""
Connects to `NLopt.jl` as the optimization backend.
Only usable if `NLopt.jl` is loaded in the current Julia session!

# Constructor

    SemOptimizerNLopt(;
        algorithm = :LD_LBFGS,
        options = Dict{Symbol, Any}(),
        local_algorithm = nothing,
        local_options = Dict{Symbol, Any}(),
        equality_constraints = Vector{NLoptConstraint}(),
        inequality_constraints = Vector{NLoptConstraint}(),
        kwargs...)

# Arguments
- `algorithm`: optimization algorithm.
- `options::Dict{Symbol, Any}`: options for the optimization algorithm
- `local_algorithm`: local optimization algorithm
- `local_options::Dict{Symbol, Any}`: options for the local optimization algorithm
- `equality_constraints::Vector{NLoptConstraint}`: vector of equality constraints
- `inequality_constraints::Vector{NLoptConstraint}`: vector of inequality constraints

# Example
```julia
my_optimizer = SemOptimizerNLopt()

# constrained optimization with augmented lagrangian
my_constrained_optimizer = SemOptimizerNLopt(;
    algorithm = :AUGLAG,
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6),
    inequality_constraints = NLoptConstraint(;f = my_constraint, tol = 0.0),
)
```

# Usage
All algorithms and options from the NLopt library are available, for more information see
the NLopt.jl package and the NLopt online documentation.
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
struct SemOptimizerNLopt{A, A2, B, B2, C} <: SemOptimizer{:NLopt}
    algorithm::A
    local_algorithm::A2
    options::B
    local_options::B2
    equality_constraints::C
    inequality_constraints::C
end