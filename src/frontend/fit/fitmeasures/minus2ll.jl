#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
minus2ll(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    minus2ll(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# SemML -----------------------------------------------------------------------------
minus2ll(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) =
    n_obs(obs)*(sem_fit.minimum + log(2π)obs.n_man)

# compute likelihood for missing data - H0 -------------------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(sem_fit::SemFit, observed, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemFIML)
    F = sem_fit.minimum
    F *= n_obs(observed)
    F += sum(log(2π)*observed.pattern_n_obs.*observed.pattern_nvar_obs)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObsMissing)
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