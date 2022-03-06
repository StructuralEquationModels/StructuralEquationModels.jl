#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
minus2ll(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    minus2ll(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM(Symbolic) + SemML
minus2ll(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) = 
    minus2ll(sem_fit.minimum, obs.n_obs, obs.n_man)

function minus2ll(minimum, n_obs, n_man)
    m2ll = n_obs*(minimum + log(2π)n_man)
    return m2ll
end