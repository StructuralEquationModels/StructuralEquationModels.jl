############################################################################
### get number of parameters
############################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
npar(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = 
    npar(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
npar(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) =
    npar(imp)

npar(imp::Union{RAM, RAMSymbolic}) = length(imp.start_val)

############################################################################
### based on χ² - 2df
############################################################################

# AICχ²(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) - 2df(sem_fit)

############################################################################
### based on χ² + 2q
############################################################################

# AICq(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) + 2length(sem_fit.model.imply.start_val)
