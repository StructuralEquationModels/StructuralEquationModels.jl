"""
    minus2ll(sem_fit::SemFit)

Return the negative 2* log likelihood.
"""
function minus2ll end

############################################################################################
# Single Models
############################################################################################

# SemFit splices loss functions ------------------------------------------------------------
minus2ll(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    minus2ll(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.optimizer,
        sem_fit.model.loss.functions...
        )

minus2ll(sem_fit::SemFit, obs, imp, optimizer, args...) = minus2ll(sem_fit.minimum, obs, imp, optimizer, args...)

# SemML ------------------------------------------------------------------------------------
minus2ll(minimum::Number, obs, imp::Union{RAM, RAMSymbolic}, optimizer, loss_ml::SemML) =
    n_obs(obs)*(minimum + log(2π)*n_man(obs))

# WLS --------------------------------------------------------------------------------------
minus2ll(minimum::Number, obs, imp::Union{RAM, RAMSymbolic}, optimizer, loss_ml::SemWLS) =
    missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(minimum::Number, observed, imp::Union{RAM, RAMSymbolic}, optimizer, loss_ml::SemFIML)
    F = minimum * n_obs(observed)
    F += log(2π)*sum(pat -> n_obs(pat)*nobserved_vars(pat), observed.patterns)
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

minus2ll(minimum, model::AbstractSemSingle) = 
    minus2ll(minimum, model.observed, model.imply, model.optimizer, model.loss.functions...)

function minus2ll(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O})
    m2ll = 0.0
    for sem in sem_fit.model.sems
        minimum = objective!(sem, sem_fit.solution)
        m2ll += minus2ll(minimum, sem)
    end
    return m2ll
end