############################################################################
### based on -2ll
############################################################################

BIC(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    minus2ll(sem_fit) + log(nobs(sem_fit))*npar(sem_fit)