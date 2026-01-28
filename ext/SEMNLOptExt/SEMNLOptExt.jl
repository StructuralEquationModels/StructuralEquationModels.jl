module SEMNLOptExt

using StructuralEquationModels, NLopt

import Base.Docs: doc

SEM = StructuralEquationModels

export SemOptimizerNLopt

include("NLopt.jl")

end
