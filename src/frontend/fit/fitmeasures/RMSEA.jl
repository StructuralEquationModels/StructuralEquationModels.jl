"""
    RMSEA(sem_fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) =
    RMSEA(dof(sem_fit), χ²(sem_fit), nsamples(sem_fit))

RMSEA(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O}) =
    sqrt(length(sem_fit.model.sems)) * RMSEA(dof(sem_fit), χ²(sem_fit), nsamples(sem_fit))

function RMSEA(dof, chi2, nsamples)
    rmsea = (chi2 - dof) / (nsamples * dof)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end
