module SEMProximalOptExt

using StructuralEquationModels
using StructuralEquationModels: print_type_name, print_field_types
using ProximalAlgorithms
using Printf

export SemOptimizerProximal

SEM = StructuralEquationModels

include("ProximalAlgorithms.jl")

end
