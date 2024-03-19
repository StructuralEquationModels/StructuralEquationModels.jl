"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSemSingle) =
    RMSEA(dof(fit), χ²(fit), nsamples(fit))

RMSEA(fit::SemFit, model::SemEnsemble) =
    sqrt(length(model.sems)) * RMSEA(dof(fit), χ²(fit), nsamples(fit))

function RMSEA(dof, chi2, nsamples)
    rmsea = (chi2 - dof) / (nsamples * dof)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end
