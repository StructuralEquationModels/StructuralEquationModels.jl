module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, SparseArrays, Symbolics,
    NLopt, FiniteDiff, ForwardDiff, PrettyTables,
    Distributions, StenoGraphs, LazyArtifacts, DelimitedFiles,
    DataFrames

import DataFrames: DataFrame
export *, ==, @StenoGraph, AbstractEdge, AbstractNode, DirectedEdge, Edge, EdgeModifier, MetaEdge, MetaNode, ModifiedEdge, ModifiedNode, Modifier, ModifyingNode, Node, NodeModifier, NodeOrEdgeModifier, SimpleNode, StenoGraphs, UndirectedEdge, convert, promote_rule, show, unarrow, unmeta, ←, →, ↔, ⇐, ⇒, ⇔

# type hierarchy
include("types.jl")
# fitted objects
include("frontend/fit/SemFit.jl")
# specification of models
include("frontend/specification/ParameterTable.jl")
include("frontend/specification/EnsembleParameterTable.jl")
include("frontend/specification/RAMMatrices.jl")
include("frontend/specification/StenoGraphs.jl")
include("frontend/fit/summary.jl")
# pretty printing
include("frontend/pretty_printing.jl")
# observed
include("observed/common.jl")
include("observed/missing.jl")
include("observed/EM.jl")
# constructor
include("frontend/specification/Sem.jl")
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
# diff
include("diff/optim.jl")
include("diff/NLopt.jl")
include("diff/Empty.jl")
# optimizer
include("optimizer/optim.jl")
include("optimizer/NLopt.jl")
# helper functions
include("additional_functions/helper.jl")
include("additional_functions/parameters.jl")
include("additional_functions/start_val/start_val.jl")
include("additional_functions/start_val/start_fabin3.jl")
include("additional_functions/start_val/start_partable.jl")
include("additional_functions/start_val/start_simple.jl")
include("additional_functions/artifacts.jl")
include("additional_functions/simulation.jl")
# identifier
include("additional_functions/identifier.jl")
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
            AbstractSemSingle, AbstractSemCollection, Sem, SemFiniteDiff, SemForwardDiff, SemEnsemble,
        SemImply, 
            RAMSymbolic, RAM, ImplyEmpty,
        start_val,
            start_fabin3, start_simple, start_parameter_table,
        SemLoss, 
            SemLossFunction, SemML, SemFIML, em_mvn, SemLasso, SemRidge,
            SemConstant, SemWLS,
        SemDiff, 
            SemDiffEmpty, SemDiffOptim, SemDiffNLopt, NLoptConstraint,
        SemObs, 
            SemObsCommon, SemObsMissing,
        sem_fit, 
        SemFit,
            minimum, solution,
        sem_summary,
        objective, objective!, gradient, gradient!, hessian, hessian!, objective_gradient!,
        ParameterTable, 
            EnsembleParameterTable, update_partable!, update_estimate!, update_start!,
            Fixed, fixed, Start, start, Label, label,
            get_identifier_indices,
        RAMMatrices, 
            RAMMatrices!,
        identifier,
        fit_measures,
            AIC, BIC, χ², df, fit_measures, minus2ll, n_par, n_obs, p_value, RMSEA, n_man,
            EmMVNModel,
        se_hessian, se_bootstrap,
        example_data,
        @StenoGraph, →, ←, ↔, ⇔
end