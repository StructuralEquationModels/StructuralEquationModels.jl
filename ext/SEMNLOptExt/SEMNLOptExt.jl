module SEMNLOptExt

using StructuralEquationModels, NLopt

SEM = StructuralEquationModels

export SemOptimizerNLopt, NLoptConstraint

include("NLopt.jl")

end
