"""
Connects to `ProximalAlgorithms.jl` as the optimization backend.

# Constructor

    SemOptimizerProximal(;
        algorithm = ProximalAlgorithms.PANOC(),
        operator_g,
        operator_h = nothing,
        kwargs...,

# Arguments
- `algorithm`: optimization algorithm.
- `operator_g`: gradient of the objective function
- `operator_h`: optional hessian of the objective function
"""
mutable struct SemOptimizerProximal{A, B, C} <: SemOptimizer{:Proximal}
    algorithm::A
    operator_g::B
    operator_h::C
end