############################################################################################
# get a map from parameters to their indices
############################################################################################

param_indices(sem_fit::SemFit) = param_indices(sem_fit.model)
param_indices(model::AbstractSemSingle) = param_indices(model.imply)
param_indices(model::SemEnsemble) = model.param_indices

############################################################################################
# construct a map from parameters to indices
############################################################################################

param_indices(ram_matrices::RAMMatrices) =
    Dict(par => i for (i, par) in enumerate(ram_matrices.params))
function param_indices(partable::ParameterTable)
    _, _, param_indices = get_par_npar_indices(partable)
    return param_indices
end

############################################################################################
# get indices of a Vector of parameter labels
############################################################################################

params_to_indices(params, param_indices::Dict{Symbol, Int}) =
    [param_indices[par] for par in params]

params_to_indices(
    params,
    obj::Union{SemFit, AbstractSemSingle, SemEnsemble, SemImply},
) = params_to_indices(params, params(obj))

function params_to_indices(params, obj::Union{ParameterTable, RAMMatrices})
    @warn "You are trying to find parameter indices from a ParameterTable or RAMMatrices object. \n
           If your model contains user-defined types, this may lead to wrong results. \n
           To be on the safe side, try to reference parameters by labels or query the indices from
           the constructed model (`params_to_indices(params, model)`)." maxlog = 1
    return params_to_indices(params, params(obj))
end

############################################################################################
# documentation
############################################################################################
"""
    params_to_indices(params, model)

Returns the indices of `params`.

# Arguments
- `params::Vector{Symbol}`: parameter labels
- `model`: either a SEM or a fitted SEM

# Examples
```julia
parameter_indices = params_to_indices([:λ₁, λ₂], my_fitted_sem)

values = solution(my_fitted_sem)[parameter_indices]
```
"""
function params_to_indices end
