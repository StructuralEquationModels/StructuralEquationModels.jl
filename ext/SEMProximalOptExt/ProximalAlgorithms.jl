############################################################################################
### Types
############################################################################################
mutable struct SemOptimizerProximal{A, B, C} <: SemOptimizer{:Proximal}
    algorithm::A
    operator_g::B
    operator_h::C
end

SEM.SemOptimizer{:Proximal}(args...; kwargs...) = SemOptimizerProximal(args...; kwargs...)

SEM.optimizer_engine_doc(engine::Val{:Proximal}) = doc(SemOptimizerProximal)

"""
Connects to `ProximalAlgorithms.jl` as the optimization backend. For more information on 
the available algorithms and options, see the online docs on [Regularization](@ref) and
the documentation of [*ProximalAlgorithms.jl*](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) / [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).

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

mutable struct ProximalResult
    result::Any
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
        solution, iterations =
            optim.algorithm(x0 = start_params, f = model, g = optim.operator_g)
    else
        solution, iterations = optim.algorithm(
            x0 = start_params,
            f = model,
            g = optim.operator_g,
            h = optim.operator_h,
        )
    end

    minimum = objective!(model, solution)

    optimization_result = Dict(
        :minimum => minimum,
        :iterations => iterations,
        :algorithm => optim.algorithm,
        :operator_g => optim.operator_g,
    )

    isnothing(optim.operator_h) ||
        push!(optimization_result, :operator_h => optim.operator_h)

    return SemFit(
        minimum,
        solution,
        start_params,
        model,
        optim,
        ProximalResult(optimization_result),
    )
end

############################################################################################
### additional methods
############################################################################################

SEM.algorithm_name(res::ProximalResult) = SEM.algorithm_name(res.result[:algorithm])
SEM.algorithm_name(::ProximalAlgorithms.IterativeAlgorithm{I,H,S,D,K}) where 
    {I, H, S, D, K} = nameof(I)

SEM.convergence(::ProximalResult) = "No standard convergence criteria for proximal \n algorithms available."
SEM.n_iterations(res::ProximalResult) = res.result[:iterations]

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, struct_inst::SemOptimizerProximal)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

function Base.show(io::IO, result::ProximalResult)
    print(io, "Minimum:          $(round(result.result[:minimum]; digits = 2)) \n")
    print(io, "No. evaluations:  $(result.result[:iterations]) \n")
    print(io, "Operator:         $(nameof(typeof(result.result[:operator_g]))) \n")
    if haskey(result.result, :operator_h)
        print(io, "Second Operator:  $(nameof(typeof(result.result[:operator_h]))) \n")
    end
end
