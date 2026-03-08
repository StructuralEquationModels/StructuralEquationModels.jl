"""
    minus2ll(fit::SemFit)

Calculate the *-2⋅log(likelihood(fit))*.

# See also
[`fit_measures`](@ref)
"""
minus2ll(fit::SemFit) = minus2ll(fit, fit.model)

############################################################################################
# Single Models
############################################################################################

function minus2ll(fit::SemFit, model::AbstractSemSingle)
    check_single_lossfun(model; throw_error = true)
    F = objective(model, fit.solution)
    return minus2ll(model.loss.functions[1], F, model)
end

# SemML ------------------------------------------------------------------------------------
function minus2ll(::SemML, F, model::AbstractSemSingle)
    return nsamples(model) * (F + log(2π) * nobserved_vars(model))
end

# WLS --------------------------------------------------------------------------------------
minus2ll(::SemWLS, F, ::AbstractSemSingle) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ*mᵢ)) + F*n
function minus2ll(::SemFIML, F, model::AbstractSemSingle)
    obs = observed(model)::SemObservedMissing
    F *= nsamples(obs)
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
# Collection
############################################################################################

function minus2ll(fit::SemFit, model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    return sum(Base.Fix1(minus2ll, fit), model.sems)
end
