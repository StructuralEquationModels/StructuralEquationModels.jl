############################################################################################
### get number of parameters
############################################################################################
"""
    n_par(sem_fit::SemFit)
    n_par(model::AbstractSemSingle)
    n_par(model::SemEnsemble)
    n_par(param_indices::Dict)

Return the number of parameters.
"""
function n_par end

n_par(fit::SemFit) = n_par(fit.model)

n_par(model::AbstractSemSingle) = n_par(model.imply)

n_par(model::SemEnsemble) = n_par(model.param_indices)

n_par(param_indices::Dict) = length(param_indices)
