"""
    minus2ll(sem_fit::SemFit)

Return the negative 2* log likelihood.
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
    # fit EM-based mean and cov if not yet fitted
    # FIXME EM could be very computationally expensive
    observed.em_model.fitted || em_mvn!(observed)

    Σ = observed.em_model.Σ
    μ = observed.em_model.μ

    F = sum(observed.patterns) do pat
        # implied covariance/mean
        Σᵢ = Σ[pat.measured_mask, pat.measured_mask]
        Σᵢ_chol = cholesky!(Σᵢ)
        ld = logdet(Σᵢ_chol)
        Σᵢ⁻¹ = LinearAlgebra.inv!(Σᵢ_chol)
        meandiffᵢ = pat.measured_mean - μ[pat.measured_mask]

        F_one_pattern(meandiffᵢ, Σᵢ⁻¹, pat.measured_cov, ld, nsamples(pat))
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
