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
