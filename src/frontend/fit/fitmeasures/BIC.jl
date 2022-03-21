############################################################################
### based on -2ll
############################################################################

BIC(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    minus2ll(sem_fit) + log(n_obs(sem_fit))*n_par(sem_fit)