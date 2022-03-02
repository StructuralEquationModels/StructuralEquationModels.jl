# SemFit splices loss functions ---------------------------------------------------------------------
p_value(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = 
    p_value(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
p_value(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemML) = p_value(sem_fit.minimum, obs, imp)
p_value(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemML) = p_value(sem_fit.minimum, obs, imp)

function p_value(minimum, observed::SemObsCommon, imply)
    chi2 = χ²(minimum, observed)
    dist_chi2 = Chisq(df(observed, imply))
    return 1 - cdf(dist_chi2, chi2)
end
