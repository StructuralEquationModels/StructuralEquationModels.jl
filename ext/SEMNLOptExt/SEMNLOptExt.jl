module SEMNLOptExt

using StructuralEquationModels, NLopt

SEM = StructuralEquationModels

export SemOptimizerNLopt

include("NLopt.jl")

end
