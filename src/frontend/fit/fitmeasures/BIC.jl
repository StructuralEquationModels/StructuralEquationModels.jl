"""
    BIC(sem_fit::SemFit)

Return the bayesian information criterion.
"""
BIC(sem_fit::SemFit) = minus2ll(sem_fit) + log(n_obs(sem_fit))*n_par(sem_fit)