fit_measures(fit) = fit_measures(fit, nparams, dof, AIC, BIC, RMSEA, χ², p_value, minus2ll)

fit_measures(fit, measures...) = Dict(Symbol(fn) => fn(fit) for fn in measures)

"""
    fit_measures(fit::SemFit, measures...) -> Dict{Symbol}

Calculate fit measures for the SEM solution.

The `measures` are the functions that calculate fit measures for a given SEM solution
([`SemFit`](@ref) object). If no `measures` are specified, the default set of measures is used.

Returns a dictionary of the fit measures, where the keys are the function names.

# Examples

```julia
fit_measures(semfit)
fit_measures(semfit, nparams, dof, p_value)
```

# See also
[`AIC`](@ref), [`BIC`](@ref), [`RMSEA`](@ref), [`χ²`](@ref), [`p_value`](@ref), [`minus2ll`](@ref)
"""
fit_measures
