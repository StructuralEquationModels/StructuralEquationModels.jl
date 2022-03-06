# SemFit splices loss functions ---------------------------------------------------------------------
RMSEA(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    RMSEA(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
RMSEA(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) =
    RMSEA(sem_fit.minimum, obs, imp)

function RMSEA(minimum, observed, imply)
    df_ = df(observed, imply)
    rmsea = (χ²(minimum, observed) - df_)/((observed.n_obs-1)*df_)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end

############################################################################
### based on χ² - 2df
############################################################################

# AICχ²(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) - 2df(sem_fit)

############################################################################
### based on χ² + 2q
############################################################################

# AICq(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = χ²(sem_fit) + 2length(sem_fit.model.imply.start_val)
