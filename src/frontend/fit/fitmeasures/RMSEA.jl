"""
    RMSEA(sem_fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(sem_fit::SemFit) = RMSEA(sem_fit, sem_fit.model)

RMSEA(sem_fit::SemFit, model::AbstractSemSingle) =
    RMSEA(df(sem_fit), χ²(sem_fit), n_obs(sem_fit))

RMSEA(sem_fit::SemFit, model::SemEnsemble) =
    sqrt(length(model.sems))*RMSEA(df(sem_fit), χ²(sem_fit), n_obs(sem_fit))

function RMSEA(df, chi2, n_obs)
    rmsea = (chi2 - df) / (n_obs*df)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end