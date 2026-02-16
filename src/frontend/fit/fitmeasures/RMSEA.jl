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

function RMSEA(fit::SemFit, model::AbstractSemSingle)
    check_single_lossfun(model; throw_error = true)
    return RMSEA(dof(fit), χ²(fit), nsamples(fit)+rmsea_correction(model.loss.functions[1]))
end

function RMSEA(fit::SemFit, model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    n = nsamples(fit)+model.n*rmsea_correction(model.sems[1].loss.functions[1])
    return sqrt(length(model.sems)) * RMSEA(dof(fit), χ²(fit), n)
end

function RMSEA(dof, chi2, N⁻)
    rmsea = (chi2 - dof) / (N⁻ * dof)
    rmsea = rmsea > 0 ? rmsea : 0
    return sqrt(rmsea)
end

# scaling corrections
rmsea_correction(::SemFIML) = 0
rmsea_correction(::SemML) = -1
rmsea_correction(::SemWLS) = -1