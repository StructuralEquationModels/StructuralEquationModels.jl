############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need an imply part.
(For example, models that only regularize parameters.)

# Constructor

    ImplyEmpty(;specification, kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
A multigroup model with ridge regularization could be specified as a `SemEnsemble` with one
model per group and an additional model with `ImplyEmpty` and `SemRidge` for the regularization part.

# Extended help

## Interfaces
- `params(::RAMSymbolic) `-> Vector of parameter labels
- `nparams(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImply`.
"""
struct ImplyEmpty{V2} <: SemImply{NoMeanStructure, ExactHessian}
    ram_matrices::V2
end

############################################################################################
### Constructors
############################################################################################

function ImplyEmpty(; specification::SemSpecification, kwargs...)
    return ImplyEmpty(convert(RAMMatrices, specification))
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, imply::ImplyEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

params(imply::ImplyEmpty) = params(imply.ram_matrices)
nparams(imply::ImplyEmpty) = nparams(imply.ram_matrices)

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply
