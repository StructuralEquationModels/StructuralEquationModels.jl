"""
    minus2ll(sem_fit::SemFit)

Return the negative 2* log likelihood.
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
    Σ, μ = obs_cov(observed), obs_mean(observed)

    F = 0.0
    for pat in observed.patterns
        nᵢ = nsamples(pat)
        # implied covariance/mean
        Σᵢ = Symmetric(Σ[pat.measured_mask, pat.measured_mask])

        ld = logdet(Σᵢ)
        Σᵢ⁻¹ = LinearAlgebra.inv!(cholesky!(Σᵢ))
        μ_diffᵢ = pat.measured_mean - μ[pat.measured_mask]

        F_pat = ld + dot(μ_diffᵢ, Σᵢ⁻¹, μ_diffᵢ)
        if nsamples(pat) > 1
            F_pat += dot(pat.measured_cov, Σᵢ⁻¹)
        end
        F += (F_pat + log(2π) * nmeasured_vars(pat)) * nsamples(pat)
    end

    #F *= nsamples(observed)
    return F
end

############################################################################################
# Collection
############################################################################################

minus2ll(fit::SemFit, model::SemEnsemble) = sum(Base.Fix1(minus2ll, fit), model.sems)
