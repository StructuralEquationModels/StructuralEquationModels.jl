"""
    dof(fit::SemFit)
    dof(model::AbstractSem)

Get the *degrees of freedom* for the SEM model.

The *degrees of freedom* for the SEM with *N* observed variables is the difference
between the number of non-redundant elements in the observed covariance matrix
(*½N(N+1)*) and the number of model parameters, *q* ([`nparams(model)`](@ref nparams)).
If the SEM also models the observed means, the formula becomes *½N(N+1) + N - q*.

# See also
[`fit_measures`](@ref)
"""
function dof end

dof(fit::SemFit) = dof(fit.model)

dof(model::AbstractSem) = n_dp(model) - nparams(model)

function n_dp(model::AbstractSemSingle)
    nvars = nobserved_vars(model)
    ndp = 0.5(nvars^2 + nvars)
    if !isnothing(model.implied.μ)
        ndp += nvars
    end
    return ndp
end

n_dp(model::SemEnsemble) = sum(n_dp.(model.sems))
