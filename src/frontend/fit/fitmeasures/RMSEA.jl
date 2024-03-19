"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSemSingle) =
    RMSEA(df(fit), χ²(fit), nsamples(fit))

RMSEA(fit::SemFit, model::SemEnsemble) =
    sqrt(length(model.sems)) * RMSEA(df(fit), χ²(fit), nsamples(fit))

function RMSEA(df, chi2, nsamples)
    rmsea = (chi2 - df) / (nsamples * df)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end
