############################################################################
### get number of parameters
############################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
n_par(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    n_par(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
n_par(sem_fit::SemFit, obs, imply::Union{RAM, RAMSymbolic}, diff, loss) = imply.n_par

############################################################################
### based on χ² - 2df
############################################################################

# AICχ²(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) - 2df(sem_fit)

############################################################################
### based on χ² + 2q
############################################################################

# AICq(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) + 2length(sem_fit.model.imply.start_val)
