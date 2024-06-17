module SEMNLOptExt

using StructuralEquationModels, NLopt

SEM = StructuralEquationModels

export SemOptimizerNLopt, NLoptConstraint

include("diff/NLopt.jl")
include("optimizer/NLopt.jl")

end
