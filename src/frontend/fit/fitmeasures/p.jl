"""
    p_value(fit::SemFit)

Calculate the *p*-value for the *χ²* test statistic.

# See also
[`fit_measures`](@ref), [`χ²`](@ref)
"""
p_value(fit::SemFit) = ccdf(Chisq(dof(fit)), χ²(fit))
