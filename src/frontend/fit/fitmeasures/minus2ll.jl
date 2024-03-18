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
    minimum = objective!(model, fit.solution)
    return minus2ll(minimum, model)
end

minus2ll(minimum::Number, model::AbstractSemSingle) =
    sum(loss -> minus2ll(minimum, model, loss), model.loss.functions)

# SemML ------------------------------------------------------------------------------------
function minus2ll(minimum::Number, model::AbstractSemSingle, loss_ml::SemML)
    obs = observed(model)
    return n_obs(obs)*(minimum + log(2π)*n_man(obs))
end

# WLS --------------------------------------------------------------------------------------
minus2ll(minimum::Number, model::AbstractSemSingle, loss_ml::SemWLS) =
    missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(minimum::Number, model::AbstractSemSingle, loss_ml::SemFIML)
    obs = observed(model)::SemObservedMissing
    F = minimum * n_obs(obs)
    F += log(2π)*sum(pat -> n_obs(pat)*nobserved_vars(pat), obs.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    if observed.em_model.fitted
        minus2ll(
            observed.em_model.μ,
            observed.em_model.Σ,
            observed.n_obs,
            observed.rows,
            observed.patterns,
            observed.obs_mean,
            observed.obs_cov,
            observed.pattern_n_obs,
            observed.pattern_nvar_obs)
    else
        em_mvn(observed)
        minus2ll(
            observed.em_model.μ,
            observed.em_model.Σ,
            observed.n_obs,
            observed.rows,
            observed.patterns,
            observed.obs_mean,
            observed.obs_cov,
            observed.pattern_n_obs,
            observed.pattern_nvar_obs)
    end
end

function minus2ll(μ, Σ, N, rows, patterns, obs_mean, obs_cov, pattern_n_obs, pattern_nvar_obs)

    F = 0.0

    for i in 1:length(rows)

        nᵢ = pattern_n_obs[i]
        # missing pattern
        pattern = patterns[i]
        # observed data
        Sᵢ = obs_cov[i]

        # implied covariance/mean
        Σᵢ = Σ[pattern, pattern]
        ld = logdet(Σᵢ)
        Σᵢ⁻¹ = inv(cholesky(Σᵢ))
        meandiffᵢ = obs_mean[i] - μ[pattern]

        F += F_one_pattern(meandiffᵢ, Σᵢ⁻¹, Sᵢ, ld, nᵢ)
        
    end

    F += sum(log(2π)*pattern_n_obs.*pattern_nvar_obs)
    #F *= N

    return F

end

############################################################################################
# Collection
############################################################################################

minus2ll(fit::SemFit, model::SemEnsemble) = sum(Base.Fix1(minus2ll, fit), model.sems)
