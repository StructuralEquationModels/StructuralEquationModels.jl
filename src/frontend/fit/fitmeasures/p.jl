# SemFit splices loss functions ---------------------------------------------------------------------
p_value(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    p_value(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
p_value(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::Union{SemML, SemFIML, SemWLS}) = 
    1 - cdf(Chisq(df(sem_fit)), χ²(sem_fit))