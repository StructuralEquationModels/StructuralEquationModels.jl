"""
    minus2ll(fit::SemFit)

Calculate the *-2⋅log(likelihood(fit))*.

# See also
[`fit_measures`](@ref)
"""
minus2ll(fit::SemFit) = minus2ll(fit.model, fit)

############################################################################################
# Single SEM Terms Models
############################################################################################

function minus2ll(term::SemLoss, fit::SemFit)
    minimum = objective(term, fit.solution)
    return minus2ll(term, minimum)
end

minus2ll(term::SemML, minimum::Number) =
    nsamples(term) * (minimum + log(2π) * nobserved_vars(term))

minus2ll(term::SemWLS, minimum::Number) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(term::SemFIML, minimum::Number)
    obs = observed(term)::SemObservedMissing
    F = minimum * nsamples(obs)
    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), obs.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    Σ, μ = obs_cov(observed), obs_mean(observed)

    # FIXME: this code is duplicate to objective(fiml, ...)
    F = sum(observed.patterns) do pat
        # implied covariance/mean
        Σᵢ = Σ[pat.measured_mask, pat.measured_mask]
        Σᵢ_chol = cholesky!(Σᵢ)
        ld = logdet(Σᵢ_chol)
        Σᵢ⁻¹ = LinearAlgebra.inv!(Σᵢ_chol)
        μ_diffᵢ = pat.measured_mean - μ[pat.measured_mask]

        F_pat = ld + dot(μ_diffᵢ, Σᵢ⁻¹, μ_diffᵢ)
        if nsamples(pat) > 1
            F_pat += dot(pat.measured_cov, Σᵢ⁻¹)
        end
        F_pat * nsamples(pat)
    end

    F += log(2π) * sum(pat -> nsamples(pat) * nmeasured_vars(pat), observed.patterns)

    return F
end

############################################################################################
# Multi-group
############################################################################################

function minus2ll(model::AbstractSem, fit::SemFit)
    check_single_lossfun(model; throw_error = true)
    sum(Base.Fix2(minus2ll, fit) ∘ _unwrap ∘ loss, sem_terms(model))
end
