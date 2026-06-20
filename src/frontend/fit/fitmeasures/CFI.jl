"""
    (1) CFI(fit::SemFit, fit_baseline::SemFit)

    (2) CFI(fit::SemFit)

Calculate the Comparative Fit Index (CFI).

The CFI ranges from 0-1 and measures how much better the model
fits the data compared to a baseline model.
If no baseline model is provided, a model with unconstrained
variances (and means) is compaired against.
For multigroup models, variances (and means) per group are free
without any equality constraints between groups.
"""
function CFI end

# if the user provides a baseline model
CFI(fit::SemFit, fit_baseline::SemFit) =
    CFI(χ²(fit), dof(fit), χ²(fit_baseline), dof(fit_baseline))

# no baseline -> variance only model
CFI(fit::SemFit) = CFI(fit, fit.model)

function CFI(fit::SemFit, model::AbstractSem)
    dof₀ = dof_varonly(model)
    χ²₀ = χ²_varonly(model)
    return CFI(χ²(fit), dof(fit), χ²₀, dof₀)
end

# basic CFI function
function CFI(χ², dof, χ²₀, dof₀)
    λ = χ² - dof
    λ₀ = χ²₀ - dof₀
    return 1 - maximum([λ, 0])/maximum([λ, λ₀, 0])
end

###
function χ²_varonly(model::AbstractSem)
    check_same_semterm_type(model; throw_error = true)
    return sum(sem_terms(model)) do semterm
        χ²_varonly(_unwrap(loss(semterm)))
    end
end

function χ²_varonly(loss::SemML)
    N⁻ = (nsamples(loss) - 1)
    S = obs_cov(observed(loss))
    Σ₀ = Diagonal(S)
    p = nobserved_vars(loss)
    return N⁻*(logdet(Σ₀) + tr(inv(Σ₀)*S) - logdet(S) - p)
end

# for the optimal variance only model, we have to solve 1/2 tr((I-XS⁻¹)^2) with X diagonal
function χ²_varonly(loss::SemWLS)
    N⁻ = (nsamples(loss) - 1)
    S⁻¹ = inv((obs_cov(observed(loss))))
    Σ₀ = Diagonal(inv(S⁻¹ .* S⁻¹)*diag(S⁻¹))
    return N⁻*0.5*tr((I - Σ₀*S⁻¹)^2)
end

# For FIML, the variance-only baseline cannot be derived automatically, so the CFI is
# `missing` unless an explicit baseline model is passed via `CFI(fit, fit_baseline)`.
# Returning `missing` (instead of throwing) keeps `fit_measures()` usable for FIML models.
χ²_varonly(loss::SemFIML) = missing

function dof_varonly(model::AbstractSem)
    return sum(sem_terms(model)) do semterm
        nparams_varonly = nobserved_vars(semterm)
        if MeanStruct(implied(semterm)) === HasMeanStruct
            nparams_varonly *= 2
        end
        return n_dp(loss(semterm)) - nparams_varonly
    end
end
