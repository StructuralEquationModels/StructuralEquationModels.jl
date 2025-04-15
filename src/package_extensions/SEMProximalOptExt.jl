"""
Connects to `ProximalAlgorithms.jl` as the optimization backend.
Can be used for regularized SEM, for a tutorial see the online docs on [Regularization](@ref).

# Constructor

    SemOptimizerProximal(;
        algorithm = ProximalAlgorithms.PANOC(),
        operator_g,
        operator_h = nothing,
        kwargs...,

# Arguments
- `algorithm`: optimization algorithm.
- `operator_g`: proximal operator (e.g., regularization penalty)
- `operator_h`: optional second proximal operator

# Usage
All algorithms and operators from `ProximalAlgorithms.jl` are available,
for more information see the online docs on [Regularization](@ref) and
the documentation of `ProximalAlgorithms.jl` / `ProximalOperators.jl`.
"""
mutable struct SemOptimizerProximal{A, B, C} <: SemOptimizer{:Proximal}
    algorithm::A
    operator_g::B
    operator_h::C
end
