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
- `identifier(::RAMSymbolic) `-> Dict containing the parameter labels and their position
- `nparams(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImply`.
"""
struct ImplyEmpty{V, V2} <: SemImply{NoMeanStructure,ExactHessian}
    identifier::V2
    n_par::V
end

############################################################################################
### Constructors
############################################################################################

function ImplyEmpty(;
        specification,
        kwargs...)

        ram_matrices = RAMMatrices(specification)
        identifier = StructuralEquationModels.identifier(ram_matrices)
        
        n_par = length(ram_matrices.parameters)

        return ImplyEmpty(identifier, n_par)
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, imply::ImplyEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

identifier(imply::ImplyEmpty) = imply.identifier
nparams(imply::ImplyEmpty) = imply.nparams

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply