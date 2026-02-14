############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need an implied part.
(For example, models that only regularize parameters.)

# Constructor

    ImpliedEmpty(;specification, kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
A multigroup model with ridge regularization could be specified as a `SemEnsemble` with one
model per group and an additional model with `ImpliedEmpty` and `SemRidge` for the regularization part.

# Extended help

## Interfaces
- `param_labels(::ImpliedEmpty) `-> Vector of parameter labels
- `nparams(::ImpliedEmpty)` -> Number of parameters
"""
struct ImpliedEmpty{A, B, C} <: SemImplied
    hessianeval::A
    meanstruct::B
    ram_matrices::C
end

############################################################################################
### Constructors
############################################################################################

function ImpliedEmpty(;
    specification,
    meanstruct = NoMeanStruct(),
    hessianeval = ExactHessian(),
    kwargs...,
)
    return ImpliedEmpty(hessianeval, meanstruct, convert(RAMMatrices, specification))
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, implied::ImpliedEmpty, par) = nothing

############################################################################################
### Recommended methods
############################################################################################

update_observed(implied::ImpliedEmpty, observed::SemObserved; kwargs...) = implied
