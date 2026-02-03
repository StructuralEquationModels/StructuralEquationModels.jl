############################################################################################
### Types
############################################################################################
mutable struct SemOptimizerProximal{A, B, C} <: SemOptimizer{:Proximal}
    algorithm::A
    operator_g::B
    operator_h::C
end

SEM.SemOptimizer{:Proximal}(args...; kwargs...) = SemOptimizerProximal(args...; kwargs...)

SEM.SemOptimizer_impltype(::Val{:Proximal}) = SemOptimizerProximal

"""
    SemOptimizerProximal(;
        algorithm = ProximalAlgorithms.PANOC(),
        operator_g,
        operator_h = nothing,
        kwargs...,
    )

Connects to `ProximalAlgorithms.jl` as the optimization backend. For more information on
the available algorithms and options, see the online docs on [Regularization](@ref) and
the documentation of [*ProximalAlgorithms.jl*](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) /
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).

# Arguments
- `algorithm`: proximal optimization algorithm.
- `operator_g`: proximal operator (e.g., regularization penalty)
- `operator_h`: optional second proximal operator
"""
SemOptimizerProximal(;
    algorithm = ProximalAlgorithms.PANOC(),
    operator_g,
    operator_h = nothing,
    kwargs...,
) = SemOptimizerProximal(algorithm, operator_g, operator_h)

############################################################################################
### Recommended methods
############################################################################################

SEM.update_observed(optimizer::SemOptimizerProximal, observed::SemObserved; kwargs...) =
    optimizer

############################################################################
### Model fitting
############################################################################

# wrapper for the Proximal optimization result
struct ProximalResult{O <: SemOptimizer{:Proximal}} <: SEM.SemOptimizerResult{O}
    optimizer::O
    n_iterations::Int
end

## connect to ProximalAlgorithms.jl
function ProximalAlgorithms.value_and_gradient(model::AbstractSem, params)
    grad = similar(params)
    obj = SEM.evaluate!(zero(eltype(params)), grad, nothing, model, params)
    return obj, grad
end

function SEM.fit(
    optim::SemOptimizerProximal,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...,
)
    if isnothing(optim.operator_h)
        solution, niterations =
            optim.algorithm(x0 = start_params, f = model, g = optim.operator_g)
    else
        solution, niterations = optim.algorithm(
            x0 = start_params,
            f = model,
            g = optim.operator_g,
            h = optim.operator_h,
        )
    end

    return SemFit(
        objective!(model, solution), # minimum
        solution,
        start_params,
        model,
        ProximalResult(optim, niterations),
    )
end

############################################################################################
### additional methods
############################################################################################

SEM.algorithm_name(res::ProximalResult) = SEM.algorithm_name(res.optimizer.algorithm)
SEM.algorithm_name(
    ::ProximalAlgorithms.IterativeAlgorithm{I, H, S, D, K},
) where {I, H, S, D, K} = nameof(I)

SEM.convergence(
    ::ProximalResult,
) = "No standard convergence criteria for proximal \n algorithms available."
SEM.n_iterations(res::ProximalResult) = res.n_iterations

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, struct_inst::SemOptimizerProximal)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

function Base.show(io::IO, result::ProximalResult)
    print(io, "No. evaluations:  $(result.n_iterations) \n")
    print(io, "Operator:         $(nameof(typeof(result.optimizer.operator_g))) \n")
    op_h = result.optimizer.operator_h
    isnothing(op_h) || print(io, "Second Operator:  $(nameof(typeof(op_h))) \n")
end
