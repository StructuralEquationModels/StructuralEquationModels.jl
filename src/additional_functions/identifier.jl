############################################################################################
# get parameter identifier
############################################################################################

identifier(sem_fit::SemFit) = identifier(sem_fit.model)
identifier(model::AbstractSemSingle) = identifier(model.imply)
identifier(model::SemEnsemble) = model.identifier

############################################################################################
# construct identifier
############################################################################################

identifier(ram_matrices::RAMMatrices) =
    Dict{Symbol, Int64}(ram_matrices.parameters .=> 1:length(ram_matrices.parameters))
function identifier(partable::ParameterTable)
    _, _, identifier = get_par_npar_identifier(partable)
    return identifier
end

############################################################################################
# get indices of a Vector of parameter labels
############################################################################################

get_identifier_indices(parameters, identifier::Dict{Symbol, Int}) =
    [identifier[par] for par in parameters]

get_identifier_indices(
    parameters,
    obj::Union{SemFit, AbstractSemSingle, SemEnsemble, SemImply},
) = get_identifier_indices(parameters, identifier(obj))

function get_identifier_indices(parameters, obj::Union{ParameterTable, RAMMatrices})
    @warn "You are trying to find parameter indices from a ParameterTable or RAMMatrices object. \n
           If your model contains user-defined types, this may lead to wrong results. \n
           To be on the safe side, try to reference parameters by labels or query the indices from 
           the constructed model (`get_identifier_indices(parameters, model)`)." maxlog = 1
    return get_identifier_indices(parameters, identifier(obj))
end

############################################################################################
# documentation
############################################################################################
"""
    get_identifier_indices(parameters, model)

Returns the indices of `parameters`.

# Arguments
- `parameters::Vector{Symbol}`: parameter labels
- `model`: either a SEM or a fitted SEM

# Examples
```julia
parameter_indices = get_identifier_indices([:λ₁, λ₂], my_fitted_sem)

values = solution(my_fitted_sem)[parameter_indices]
```
"""
function get_identifier_indices end
