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
