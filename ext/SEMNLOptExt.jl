module SEMNLOptExt

using StructuralEquationModels, NLopt

SEM = StructuralEquationModels

export SemOptimizerNLopt, NLoptConstraint

include("optimizer/NLopt.jl")

end
