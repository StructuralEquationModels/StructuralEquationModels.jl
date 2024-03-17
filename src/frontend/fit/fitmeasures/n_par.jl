############################################################################################
### get number of parameters
############################################################################################
"""
    nparams(sem_fit::SemFit)
    nparams(model::AbstractSemSingle)
    nparams(model::SemEnsemble)
    nparams(identifier::Dict)

Return the number of parameters.
"""
function nparams end

nparams(fit::SemFit) = nparams(fit.model)

nparams(model::AbstractSemSingle) = nparams(model.imply)

nparams(model::SemEnsemble) = nparams(model.identifier)

nparams(identifier::Dict) = length(identifier)
