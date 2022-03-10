nobs(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    nobs(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
nobs(sem_fit::SemFit, obs::Union{SemObsCommon, SemObsMissing}, imp, diff, loss_ml) =
    obs.n_obs
