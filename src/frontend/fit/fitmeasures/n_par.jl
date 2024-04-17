############################################################################################
### get number of parameters
############################################################################################
"""
    nparams(sem_fit::SemFit)
    nparams(model::AbstractSem)

Return the number of parameters.
"""
nparams(obj::Any) = length(params(obj))
