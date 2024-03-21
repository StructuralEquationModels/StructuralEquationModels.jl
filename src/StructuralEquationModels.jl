module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, StatsBase, SparseArrays, Symbolics,
    NLopt, FiniteDiff, PrettyTables,
    Distributions, StenoGraphs, LazyArtifacts, DelimitedFiles,
    DataFrames

export StenoGraphs, @StenoGraph, meld

const SEM = StructuralEquationModels

# type hierarchy
include("types.jl")
include("objective_gradient_hessian.jl")
# fitted objects
include("frontend/fit/SemFit.jl")
# specification of models
include("additional_functions/params_array.jl")
include("frontend/specification/ParameterTable.jl")
include("frontend/specification/RAMMatrices.jl")
include("frontend/specification/EnsembleParameterTable.jl")
include("frontend/specification/StenoGraphs.jl")
include("frontend/fit/summary.jl")
# pretty printing
include("frontend/pretty_printing.jl")
# observed
include("observed/data.jl")
include("observed/covariance.jl")
include("observed/missing.jl")
include("observed/EM.jl")
# constructor
include("frontend/specification/Sem.jl")
include("frontend/specification/documentation.jl")
# imply
include("imply/RAM/symbolic.jl")
include("imply/RAM/generic.jl")
include("imply/empty.jl")
# loss
include("loss/ML/ML.jl")
include("loss/ML/FIML.jl")
include("loss/regularization/ridge.jl")
include("loss/WLS/WLS.jl")
include("loss/constant/constant.jl")
# optimizer
include("diff/optim.jl")
include("diff/NLopt.jl")
include("diff/Empty.jl")
# optimizer
include("optimizer/documentation.jl")
include("optimizer/optim.jl")
include("optimizer/NLopt.jl")
# helper functions
include("additional_functions/helper.jl")
include("additional_functions/start_val/start_val.jl")
include("additional_functions/start_val/start_fabin3.jl")
include("additional_functions/start_val/start_partable.jl")
include("additional_functions/start_val/start_simple.jl")
include("additional_functions/artifacts.jl")
include("additional_functions/simulation.jl")
# identifier
#include("additional_functions/identifier.jl")
# fit measures
include("frontend/fit/fitmeasures/AIC.jl")
include("frontend/fit/fitmeasures/BIC.jl")
include("frontend/fit/fitmeasures/chi2.jl")
include("frontend/fit/fitmeasures/df.jl")
include("frontend/fit/fitmeasures/minus2ll.jl")
include("frontend/fit/fitmeasures/n_par.jl")
include("frontend/fit/fitmeasures/n_obs.jl")
include("frontend/fit/fitmeasures/p.jl")
include("frontend/fit/fitmeasures/RMSEA.jl")
include("frontend/fit/fitmeasures/n_man.jl")
include("frontend/fit/fitmeasures/fit_measures.jl")
# standard errors
include("frontend/fit/standard_errors/hessian.jl")
include("frontend/fit/standard_errors/bootstrap.jl")



export  AbstractSem,
            AbstractSemSingle, AbstractSemCollection, Sem, SemFiniteDiff,
            SemEnsemble,
        MeanStructure, NoMeanStructure, HasMeanStructure,
        HessianEvaluation, ExactHessian, ApproximateHessian,
        SemImply,
            RAMSymbolic, RAMSymbolicZ, RAM, ImplyEmpty, imply,
        start_val,
            start_fabin3, start_simple, start_parameter_table,
        SemLoss,
            SemLossFunction, SemML, SemFIML, em_mvn, SemLasso, SemRidge,
            SemConstant, SemWLS, loss,
        SemOptimizer,
            SemOptimizerEmpty, SemOptimizerOptim, SemOptimizerNLopt, NLoptConstraint,
            optimizer, n_iterations, convergence,
        SemObserved,
            SemObservedData, SemObservedCovariance, SemObservedMissing, observed,
        sem_fit,
        SemFit,
            minimum, solution,
        sem_summary,
        objective!, gradient!, hessian!, objective_gradient!, objective_hessian!,
            gradient_hessian!, objective_gradient_hessian!,
        ParameterTable,
            EnsembleParameterTable, update_partable!, update_estimate!, update_start!, update_se_hessian!,
            Fixed, fixed, Start, start, Label, label, sort_vars!, sort_vars,
        RAMMatrices,
            RAMMatrices!,
        params, nparams,
        fit_measures,
            AIC, BIC, χ², df, fit_measures, minus2ll, n_obs, p_value, RMSEA, n_man,
            EmMVNModel,
        se_hessian, se_bootstrap,
        example_data,
        swap_observed, update_observed,
        @StenoGraph, →, ←, ↔, ⇔
end