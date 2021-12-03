module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, SparseArrays, Symbolics,
    NLopt, FiniteDiff, ForwardDiff

include("types.jl")
include("observed/common.jl")
include("observed/missing.jl")
include("additional_functions/helper.jl")
include("imply/RAM.jl")
include("loss/ML/ML.jl")
include("loss/ML/FIML.jl")
include("loss/regularization/ridge.jl")
include("loss/GLS/GLS.jl")
# include("loss/fiml.jl")
# include("loss/definition_variables.jl")
# include("loss/constant.jl")
# include("multigroup.jl")
include("diff/optim.jl")
include("diff/NLopt.jl")
include("optimizer/optim.jl")
include("optimizer/NLopt.jl")
include("additional_functions/start_val.jl")
# include("optimizer/nlopt.jl")

export  AbstractSem, 
            Sem, SemFiniteDiff, SemForwardDiff, SemEnsemble,
        SemImply, 
            RAMSymbolic,
        start_fabin3, start_simple,
        SemLoss, 
            SemLossFunction, SemML, SemFIML, SemDefinition, SemLasso, SemRidge,
            SemConstant, SemWLS, SemRidge,
        SemDiff, 
            SemDiffOptim, SemDiffNLopt,
        SemObs, 
            SemObsCommon, SemObsMissing,
        sem_fit,
        objective, objective!, gradient, gradient!, hessian, hessian!

end # module
