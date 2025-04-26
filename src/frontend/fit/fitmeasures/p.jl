"""
    p(sem_fit::SemFit)

Return the p value computed from the χ² test statistic.
"""
p_value(sem_fit::SemFit) = ccdf(Chisq(dof(sem_fit)), χ²(sem_fit))
