"""
    χ²(fit::SemFit)

Calculate the *χ²* (*chi-square*) value for the `fit`.

The *χ²* is a test statistic for the SEM goodness-of-fit.
It compares the *implied* covariance matrix of the SEM model
with the *observed* covariance matrix.

# See also
[`fit_measures`](@ref)
"""
χ²(fit::SemFit) = χ²(fit, fit.model)

function χ²(fit::SemFit, model::AbstractSem)
    terms = sem_terms(model)
    @assert !isempty(terms)

    L = check_same_semterm_type(model; throw_error = true)
    return χ²(L, fit, model)
end

# bollen, p. 115, only correct for GLS weight matrix
χ²(::Type{<:SemWLS}, fit::SemFit, model::AbstractSem) = (nsamples(model) - 1) * fit.minimum

function χ²(::Type{<:SemML}, fit::SemFit, model::AbstractSem)
    G = sum(loss_terms(model)) do term
        if issemloss(term)
            data = observed(term)
            something(weight(term), 1.0) * (logdet(obs_cov(data)) + nobserved_vars(data))
        else
            return 0.0
        end
    end
    return (nsamples(model) - 1) * (fit.minimum - G)
end

function χ²(::Type{<:SemFIML}, fit::SemFit, model::AbstractSem)
    ll_H0 = minus2ll(fit)
    ll_H1 = sum(minus2ll ∘ observed, sem_terms(model))
    return ll_H0 - ll_H1
end
