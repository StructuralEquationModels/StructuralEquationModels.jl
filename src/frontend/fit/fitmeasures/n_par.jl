############################################################################################
### get number of parameters
############################################################################################
"""
    n_par(sem_fit::SemFit)
    n_par(model::AbstractSemSingle)
    n_par(model::SemEnsemble)
    n_par(identifier::Dict)

Return the number of parameters.
"""
function n_par end

n_par(fit::SemFit) = n_par(fit.model)

n_par(model::AbstractSemSingle) = n_par(model.imply)

n_par(model::SemEnsemble) = n_par(model.identifier)

n_par(identifier::Dict) = length(identifier)
