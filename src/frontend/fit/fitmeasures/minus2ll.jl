"""
    minus2ll(fit::SemFit)

Calculate the *-2⋅log(likelihood(fit))*.

# See also
[`fit_measures`](@ref)
"""
function minus2ll end

############################################################################################
# Single Models
############################################################################################

minus2ll(fit::SemFit) = minus2ll(fit, fit.model)

function minus2ll(fit::SemFit, model::AbstractSemSingle)
    minimum = objective(model, fit.solution)
    return minus2ll(minimum, model)
end

minus2ll(minimum::Number, model::AbstractSemSingle) =
    sum(lossfun -> minus2ll(lossfun, minimum, model), model.loss.functions)

# SemML ------------------------------------------------------------------------------------
function minus2ll(lossfun::SemML, minimum::Number, model::AbstractSemSingle)
    obs = observed(model)
    return nsamples(obs) * (minimum + log(2π) * nobserved_vars(obs))
end

# WLS --------------------------------------------------------------------------------------
minus2ll(lossfun::SemWLS, minimum::Number, model::AbstractSemSingle) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(lossfun::SemFIML, minimum::Number, model::AbstractSemSingle)
    obs = observed(model)::SemObservedMissing
    F = minimum * nsamples(obs)
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

minus2ll(fit::SemFit, model::SemEnsemble) = sum(Base.Fix1(minus2ll, fit), model.sems)
