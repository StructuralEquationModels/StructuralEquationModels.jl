module SEMProximalOptExt

using StructuralEquationModels, ProximalAlgorithms
using StructuralEquationModels: print_type_name, print_field_types

import Base.Docs: doc

export SemOptimizerProximal

SEM = StructuralEquationModels

include("ProximalAlgorithms.jl")

end
