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
- `n_par(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImply`.
"""
struct ImplyEmpty{V, V2} <: SemImply
    param_indices::V2
    n_par::V
end

############################################################################################
### Constructors
############################################################################################

function ImplyEmpty(; specification, kwargs...)
    ram_matrices = RAMMatrices(specification)

    n_par = length(ram_matrices.params)

    return ImplyEmpty(param_indices(ram_matrices), n_par)
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

param_indices(imply::ImplyEmpty) = imply.param_indices
n_par(imply::ImplyEmpty) = imply.n_par

update_observed(imply::ImplyEmpty, observed::SemObserved; kwargs...) = imply
