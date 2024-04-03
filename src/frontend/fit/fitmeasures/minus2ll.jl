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
    F += log(2π) * sum(pat -> nsamples(pat) * nobserved_vars(pat), obs.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    observed.em_model.fitted || em_mvn(observed)

    μ = observed.em_model.μ
    Σ = observed.em_model.Σ

    F = 0.0
    for pat in observed.patterns
        nᵢ = nsamples(pat)
        # implied covariance/mean
        Σᵢ = Σ[pat.obs_mask, pat.obs_mask]

        ld = logdet(Σᵢ)
        Σᵢ⁻¹ = LinearAlgebra.inv!(cholesky!(Σᵢ))
        μ_diffᵢ = pat.obs_mean - μ[pat.obs_mask]

        F_pat = ld + dot(μ_diffᵢ, Σᵢ⁻¹, μ_diffᵢ)
        if n_obs(pat) > 1
            F_pat += dot(pat.obs_cov, Σᵢ⁻¹)
        end
        F += (F_pat + log(2π) * nobserved_vars(pat)) * n_obs(pat)
    end

    #F *= nsamples(observed)
    return F
end

############################################################################################
# Collection
############################################################################################

minus2ll(fit::SemFit, model::SemEnsemble) = sum(Base.Fix1(minus2ll, fit), model.sems)
