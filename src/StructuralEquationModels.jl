module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, SparseArrays, Symbolics,
    NLopt, FiniteDiff, ForwardDiff

# type hierarchy
include("types.jl")
# observed
include("observed/common.jl")
include("observed/missing.jl")
# constructor
include("sem.jl")
# helper functions
include("additional_functions/helper.jl")
include("additional_functions/parameters.jl")
include("additional_functions/start_val.jl")
# imply
include("imply/RAM/symbolic.jl")
include("imply/RAM/generic.jl")
include("imply/empty.jl")
include("imply/SNLLS/SNLLS.jl")
# loss
include("loss/ML/ML.jl")
include("loss/ML/FIML.jl")
include("loss/regularization/ridge.jl")
include("loss/WLS/WLS.jl")
include("loss/constant/constant.jl")
include("loss/SNLLS/SNLLS.jl")
# diff
include("diff/optim.jl")
include("diff/NLopt.jl")
# optimizer
include("optimizer/optim.jl")
include("optimizer/NLopt.jl")


export  AbstractSem, 
            Sem, SemFiniteDiff, SemForwardDiff, SemEnsemble,
        SemImply, 
            RAMSymbolic, RAM, SNLLS, RAMMatrices,
        start_fabin3, start_simple,
        SemLoss, 
            SemLossFunction, SemML, SemFIML, SemDefinition, SemLasso, SemRidge,
            SemConstant, SemWLS, SemRidge, SemSNLLS,
        SemDiff, 
            SemDiffOptim, SemDiffNLopt,
        SemObs, 
            SemObsCommon, SemObsMissing,
        sem_fit,
        objective, objective!, gradient, gradient!, hessian, hessian!


end