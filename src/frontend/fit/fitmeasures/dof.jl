"""
    dof(fit::SemFit)
    dof(model::AbstractSem)

Get the *degrees of freedom* for the SEM model.

The degrees of freedom for the SEM model with *N* observed variables
is the difference between the number of parameters
required to define the *N×N* covariance matrix (*½N(N+1)*)
(plus *N* parameters for the observed means vector, if present),
and the number of model parameters, [`nparams(model)`](@ref nparams).
"""
function dof end

dof(fit::SemFit) = dof(fit.model)

dof(model::AbstractSem) = n_dp(model) - nparams(model)

# length of Σ and μ (if present)
function n_dp(implied::SemImplied)
    nvars = nobserved_vars(implied)
    ndp = 0.5(nvars^2 + nvars)
    if !isnothing(implied.μ)
        ndp += nvars
    end
    return ndp
end

n_dp(term::SemLoss) = n_dp(implied(term))

n_dp(model::AbstractSem) = sum(n_dp ∘ loss, sem_terms(model))
