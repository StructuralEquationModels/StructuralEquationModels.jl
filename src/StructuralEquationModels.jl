module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, SparseArrays, Symbolics,
    NLopt, FiniteDiff, ForwardDiff, PrettyTables,
    Distributions

# type hierarchy
include("types.jl")
# fitted objects
include("frontend/fit/SemFit.jl")
# specification of models
include("frontend/specification/ParameterTable.jl")
include("frontend/specification/parser.jl")
include("frontend/specification/RAMMatrices.jl")
# pretty printing
include("frontend/pretty_printing.jl")
# observed
include("observed/common.jl")
include("observed/missing.jl")
# constructor
include("frontend/specification/Sem.jl")
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
# fit measures
include("frontend/fit/fitmeasures/AIC.jl")
include("frontend/fit/fitmeasures/BIC.jl")
include("frontend/fit/fitmeasures/chi2.jl")
include("frontend/fit/fitmeasures/df.jl")
include("frontend/fit/fitmeasures/F.jl")
include("frontend/fit/fitmeasures/fit_measures.jl")
include("frontend/fit/fitmeasures/minus2ll.jl")
include("frontend/fit/fitmeasures/npar.jl")
include("frontend/fit/fitmeasures/p.jl")
include("frontend/fit/fitmeasures/RMSEA.jl")


export  AbstractSem, 
            AbstractSemSingle, AbstractSemCollection, Sem, SemFiniteDiff, SemForwardDiff, SemEnsemble,
        SemImply, 
            RAMSymbolic, RAM, SNLLS, ImplyEmpty,
        start_fabin3, start_simple, start_parameter_table,
        SemLoss, 
            SemLossFunction, SemML, SemFIML, em_mvn, SemLasso, SemRidge,
            SemConstant, SemWLS, SemSNLLS,
        SemDiff, 
            SemDiffOptim, SemDiffNLopt,
        SemObs, 
            SemObsCommon, SemObsMissing,
        sem_fit, SemFit,
        objective, objective!, gradient, gradient!, hessian, hessian!, objective_gradient!,
        SemSpec,
            ParameterTable, update_partable!, update_estimate!, update_start!,
            SpecEmpty,
        RAMMatrices, 
            RAMMatrices!,
        fit_measures,
            AIC, BIC, χ², df, Fₘᵢₙ, fit_measures, minus2ll, npar, p_value, RMSEA
end