# SemFit splices loss functions ---------------------------------------------------------------------
df(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    df(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
df(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) = df(obs, imp)

# generic
function df(observed::SemObsCommon, imply)
    n_par = npar(imply)
    n_dp = 0.5(observed.n_man^2 + observed.n_man)
    if !isnothing(imply.μ)
        n_dp += observed.n_man
    end
    return n_dp - n_par
end
