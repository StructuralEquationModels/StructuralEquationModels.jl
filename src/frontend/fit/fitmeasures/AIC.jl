"""
    AIC(fit::SemFit)

Calculate the *AIC* ([*Akaike information criterion*](https://en.wikipedia.org/wiki/Akaike_information_criterion)).

# See also
[`fit_measures`](@ref)
"""
AIC(fit::SemFit) = minus2ll(fit) + 2nparams(fit)
