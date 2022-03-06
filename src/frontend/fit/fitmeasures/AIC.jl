############################################################################
### based on -2ll
############################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
AIC(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    AIC(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
AIC(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) =
    minus2ll(sem_fit) + 2npar(sem_fit)

function AIC(minimum, n_man, n_obs, n_par)
    AIC = minus2ll(minimum, n_obs, n_man) + 2n_par
    return AIC
end

############################################################################
### based on χ² - 2df
############################################################################

# AICχ²(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) - 2df(sem_fit)

############################################################################
### based on χ² + 2q
############################################################################

# AICq(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) + 2length(sem_fit.model.imply.start_val)
