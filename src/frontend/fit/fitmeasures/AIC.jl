############################################################################
### based on -2ll
############################################################################
"""
    AIC(sem_fit::SemFit)

Return the akaike information criterion.
"""
AIC(sem_fit::SemFit) = minus2ll(sem_fit) + 2n_par(sem_fit)

############################################################################
### based on χ² - 2df
############################################################################

# AICχ²(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) - 2df(sem_fit)

############################################################################
### based on χ² + 2q
############################################################################

# AICq(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) + 2npar(sem_fit)