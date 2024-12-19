module SEMProximalOptExt

using StructuralEquationModels
using ProximalCore, ProximalAlgorithms, ProximalOperators

export SemOptimizerProximal

SEM = StructuralEquationModels

#ProximalCore.prox!(y, f, x, gamma) = ProximalOperators.prox!(y, f, x, gamma)

include("diff/Proximal.jl")
include("optimizer/ProximalAlgorithms.jl")

end
