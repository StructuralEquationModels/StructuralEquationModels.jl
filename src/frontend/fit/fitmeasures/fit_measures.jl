fit_measures(fit) =
    fit_measures(fit, nparams, dof, AIC, BIC, RMSEA, χ², p_value, minus2ll)

fit_measures(fit, measures...) = Dict(Symbol(fn) => fn(fit) for fn in measures)

"""
    fit_measures(fit, measures...)

Calculate fit measures for the SEM solution.

Returns a dictionary of the fit measures for the given SEM solution.
The keys are the measure names. The `measures` are functions that take SEM solution as an input.
If no `measures` are specified, the default set of measures is used.
"""
fit_measures
