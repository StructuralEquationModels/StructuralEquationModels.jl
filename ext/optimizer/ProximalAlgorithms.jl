############################################################################################
### Types
############################################################################################
"""
Connects to `ProximalAlgorithms.jl` as the optimization backend.

# Constructor

    SemOptimizerProximal(;
        algorithm = ProximalAlgorithms.PANOC(),
        options = Dict{Symbol, Any}(),
        operator_g,
        operator_h = nothing,
        kwargs...,

# Arguments
- `algorithm`: optimization algorithm.
- `options::Dict{Symbol, Any}`: options for the optimization algorithm
- `operator_g`: gradient of the objective function
- `operator_h`: optional hessian of the objective function
"""
mutable struct SemOptimizerProximal{A, B, C, D} <: SemOptimizer{:Proximal}
    algorithm::A
    options::B
    operator_g::C
    operator_h::D
end

SEM.SemOptimizer{:Proximal}(args...; kwargs...) = SemOptimizerProximal(args...; kwargs...)

SemOptimizerProximal(;
    algorithm = ProximalAlgorithms.PANOC(),
    options = Dict{Symbol, Any}(),
    operator_g,
    operator_h = nothing,
    kwargs...,
) = SemOptimizerProximal(algorithm, options, operator_g, operator_h)

############################################################################################
### Recommended methods
############################################################################################

SEM.update_observed(optimizer::SemOptimizerProximal, observed::SemObserved; kwargs...) =
    optimizer

############################################################################################
### additional methods
############################################################################################

SEM.algorithm(optimizer::SemOptimizerProximal) = optimizer.algorithm
SEM.options(optimizer::SemOptimizerProximal) = optimizer.options

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemOptimizerProximal)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

## connect do ProximalAlgorithms.jl as backend
ProximalCore.gradient!(grad, model::AbstractSem, parameters) =
    objective_gradient!(grad, model::AbstractSem, parameters)

mutable struct ProximalResult
    result::Any
end

function SEM.sem_fit(
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
        ProximalResult(optimization_result),
    )
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::ProximalResult)
    print(io, "Minimum:          $(round(result.result[:minimum]; digits = 2)) \n")
    print(io, "No. evaluations:  $(result.result[:iterations]) \n")
    print(io, "Operator:         $(nameof(typeof(result.result[:operator_g]))) \n")
    if haskey(result.result, :operator_h)
        print(io, "Second Operator:  $(nameof(typeof(result.result[:operator_h]))) \n")
    end
end
