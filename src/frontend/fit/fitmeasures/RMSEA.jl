"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSem) =
    sqrt(nsem_terms(model)) * RMSEA(dof(fit), χ²(fit), nsamples(fit))

RMSEA(dof::Number, chi2::Number, nsamples::Number) =
    sqrt(max((chi2 - dof) / (nsamples * dof), 0.0))
