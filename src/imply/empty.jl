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
- `n_par(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImply`.
"""
struct ImplyEmpty{V, V2} <: SemImply
    identifier::V2
    n_par::V
end

############################################################################################
### Constructors
############################################################################################

function ImplyEmpty(; specification, kwargs...)
    ram_matrices = RAMMatrices(specification)
    identifier = StructuralEquationModels.identifier(ram_matrices)

    n_par = length(ram_matrices.parameters)

    return ImplyEmpty(identifier, n_par)
end

############################################################################################
### methods
############################################################################################

objective!(imply::ImplyEmpty, par, model) = nothing
gradient!(imply::ImplyEmpty, par, model) = nothing
hessian!(imply::ImplyEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

identifier(imply::ImplyEmpty) = imply.identifier
n_par(imply::ImplyEmpty) = imply.n_par

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply
