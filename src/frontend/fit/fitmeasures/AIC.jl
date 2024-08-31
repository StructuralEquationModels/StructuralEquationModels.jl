"""
    AIC(fit::SemFit)

Calculate the *AIC* ([*Akaike information criterion*](https://en.wikipedia.org/wiki/Akaike_information_criterion)).
"""
AIC(fit::SemFit) = minus2ll(fit) + 2nparams(fit)
