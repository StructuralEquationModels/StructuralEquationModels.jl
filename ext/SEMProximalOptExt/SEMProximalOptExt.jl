module SEMProximalOptExt

using StructuralEquationModels
using ProximalAlgorithms
using StructuralEquationModels: print_type_name, print_field_types

export SemOptimizerProximal

SEM = StructuralEquationModels

include("ProximalAlgorithms.jl")

end
