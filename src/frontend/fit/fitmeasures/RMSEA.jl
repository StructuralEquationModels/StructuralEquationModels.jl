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
RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

# scaling corrections
RMSEA_corr_scale(::Type{<:SemFIML}) = 0
RMSEA_corr_scale(::Type{<:SemML}) = -1
RMSEA_corr_scale(::Type{<:SemWLS}) = -1

function RMSEA(fit::SemFit, model::AbstractSem)
    term_type = check_same_semterm_type(model; throw_error = true)
    n = nsamples(fit) + nsem_terms(model) * RMSEA_corr_scale(term_type)
    sqrt(nsem_terms(model)) * RMSEA(dof(fit), χ²(fit), n)
end

RMSEA(dof::Number, chi2::Number, nsamples::Number) =
    sqrt(max((chi2 - dof) / (nsamples * dof), 0.0))
