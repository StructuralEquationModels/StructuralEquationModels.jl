"""
    BIC(fit::SemFit)

Calculate the *BIC* ([*Bayesian information criterion*](https://en.wikipedia.org/wiki/Bayesian_information_criterion)).

# See also
[`fit_measures`](@ref)
"""
BIC(fit::SemFit) = minus2ll(fit) + log(nsamples(fit)) * nparams(fit)
