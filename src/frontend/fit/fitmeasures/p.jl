"""
    p_value(fit::SemFit)

Calculate the *p*-value for the *χ²* test statistic.
"""
p_value(fit::SemFit) = ccdf(Chisq(dof(fit)), χ²(fit))
