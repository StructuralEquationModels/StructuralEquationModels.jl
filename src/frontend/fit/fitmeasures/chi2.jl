#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
χ²(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    χ²(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
χ²(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemML) = χ²(sem_fit.minimum, obs)
χ²(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemML) = χ²(sem_fit.minimum, obs)


# generic SemObsCommon
χ²(minimum, observed::SemObsCommon) = (observed.n_obs-1)*Fₘᵢₙ(minimum, observed.obs_cov, observed.n_man)

#####################################################################################################
# Collections
#####################################################################################################