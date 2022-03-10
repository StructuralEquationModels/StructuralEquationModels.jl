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
function RMSEA(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::Union{SemML, SemFIML})
    df = df(sem_fit)
    rmsea = (χ²(sem_fit) - df) / ((nobs(sem_fit))*df)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end