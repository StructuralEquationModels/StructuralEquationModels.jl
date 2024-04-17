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
- `params(::RAMSymbolic) `-> Dict containing the parameter labels and their position
- `nparams(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImply`.
"""
struct ImplyEmpty <: SemImply{NoMeanStructure,ExactHessian}
    params::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

function ImplyEmpty(;
        specification::SemSpecification,
        kwargs...)

        return ImplyEmpty(params(spec))
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, imply::ImplyEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply