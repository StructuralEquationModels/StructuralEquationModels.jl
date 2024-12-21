## connect do ProximalAlgorithms.jl as backend
ProximalCore.gradient!(grad, model::AbstractSem, parameters) =
    objective_gradient!(grad, model::AbstractSem, parameters)

mutable struct ProximalResult
    result::Any
end

function SEM.sem_fit(
    model::AbstractSemSingle{O, I, L, D};
    start_val = start_val,
    kwargs...,
) where {O, I, L, D <: SemOptimizerProximal}
    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    if isnothing(model.optimizer.operator_h)
        solution, iterations = model.optimizer.algorithm(
            x0 = start_val,
            f = model,
            g = model.optimizer.operator_g,
        )
    else
        solution, iterations = model.optimizer.algorithm(
            x0 = start_val,
            f = model,
            g = model.optimizer.operator_g,
            h = model.optimizer.operator_h,
        )
    end

    minimum = objective!(model, solution)

    optimization_result = Dict(
        :minimum => minimum,
        :iterations => iterations,
        :algorithm => model.optimizer.algorithm,
        :operator_g => model.optimizer.operator_g,
    )

    isnothing(model.optimizer.operator_h) ||
        push!(optimization_result, :operator_h => model.optimizer.operator_h)

    return SemFit(minimum, solution, start_val, model, ProximalResult(optimization_result))
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
