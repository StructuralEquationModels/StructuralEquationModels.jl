"""
    RMSEA(fit::SemFit)

Calculate the RMSEA ([*Root Mean Squared Error of Approximation*](https://meth.psychopen.eu/index.php/meth/article/download/2333/2333.html?inline=1#sec1)):
``
\\mathrm{RMSEA} = \\sqrt{\\frac{\\chi^2 - N_{\\mathrm{df}}}{N_{\\mathrm{obs}} * N_{\\mathrm{df}}}},
``
where `χ²` is the chi-squared statistic, `df` is the degrees of freedom, and `N_obs` is the number of observations
for the SEM model.
"""
RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSem) =
    sqrt(nsem_terms(model)) * RMSEA(dof(fit), χ²(fit), nsamples(fit))

RMSEA(dof::Number, chi2::Number, nsamples::Number) =
    sqrt(max((chi2 - dof) / (nsamples * dof), 0.0))
