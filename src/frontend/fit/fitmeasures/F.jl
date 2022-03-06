# compute F-values

#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
Fₘᵢₙ(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    Fₘᵢₙ(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM(Symbolic) + SemML
Fₘᵢₙ(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemML) = Fₘᵢₙ(sem_fit.minimum, obs)
Fₘᵢₙ(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemML) = Fₘᵢₙ(sem_fit.minimum, obs)

# generic SemObsCommon
Fₘᵢₙ(minimum, observed::SemObsCommon) = Fₘᵢₙ(minimum, observed.obs_cov, observed.n_man)

function Fₘᵢₙ(minimum, obs_cov, n_man)
    F = minimum - logdet(obs_cov) - n_man
    return F
end

#! how for FIML?