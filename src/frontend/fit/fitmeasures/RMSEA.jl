"""
    RMSEA(fit::SemFit)

Calculate the RMSEA ([*Root Mean Squared Error of Approximation*](https://meth.psychopen.eu/index.php/meth/article/download/2333/2333.html?inline=1#sec1)).

Uses the formula
```math
\\mathrm{RMSEA} = \\sqrt{\\frac{\\chi^2 - N_{\\mathrm{df}}}{N_{\\mathrm{obs}} * N_{\\mathrm{df}}}},
```
where *χ²* is the chi-squared statistic, ``N_{\\mathrm{df}}`` is the degrees of freedom,
and ``N_{\\mathrm{obs}}`` is the (corrected) number of observations
for the SEM model.

# See also
[`fit_measures`](@ref), [`χ²`](@ref), [`dof`](@ref)

# Extended help

For multigroup models, the correction proposed by J.H. Steiger is applied
(see [Steiger, J. H. (1998). *A note on multiple sample extensions of the RMSEA fit index*](https://doi.org/10.1080/10705519809540115)).
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSemSingle) = RMSEA(dof(fit), χ²(fit), nsamples(fit))

RMSEA(fit::SemFit, model::SemEnsemble) =
    sqrt(length(model.sems)) * RMSEA(dof(fit), χ²(fit), nsamples(fit))

function RMSEA(dof, chi2, nsamples)
    rmsea = (chi2 - dof) / (nsamples * dof)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end
