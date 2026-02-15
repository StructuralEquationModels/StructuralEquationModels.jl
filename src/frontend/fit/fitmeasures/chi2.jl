"""
    χ²(fit::SemFit)

Return the χ² value.
"""
χ²(fit::SemFit) = χ²(fit, fit.model)

############################################################################################
# Single Models
############################################################################################

function χ²(fit::SemFit, model::AbstractSemSingle)
    check_single_lossfun(model; throw_error = true)
    return χ²(model.loss.functions[1], fit::SemFit, model::AbstractSemSingle)
end

χ²(::SemML, fit::SemFit, model::AbstractSemSingle) =
    (nsamples(fit) - 1) *
    (fit.minimum - logdet(obs_cov(observed(model))) - nobserved_vars(observed(model)))

# bollen, p. 115, only correct for GLS weight matrix
χ²(::SemWLS, fit::SemFit, model::AbstractSemSingle) =
    (nsamples(fit) - 1) * fit.minimum

# FIML
function χ²(::SemFIML, fit::SemFit, model::AbstractSemSingle)
    ll_H0 = minus2ll(fit)
    ll_H1 = minus2ll(observed(model))
    return ll_H0 - ll_H1
end

############################################################################################
# Collections
############################################################################################

function χ²(fit::SemFit, model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    lossfun = model.sems[1].loss.functions[1]
    return χ²(lossfun, fit, model)
end

function χ²(::SemWLS, fit::SemFit, models::SemEnsemble)
    return (nsamples(models) - models.n) * fit.minimum
end

function χ²(::SemML, fit::SemFit, models::SemEnsemble)
    G = sum(zip(models.weights, models.sems)) do (w, model)
        data = observed(model)
        w * (logdet(obs_cov(data)) + nobserved_vars(data))
    end
    return (nsamples(models) - models.n) * (fit.minimum - G)
end

function χ²(::SemFIML, fit::SemFit, models::SemEnsemble)
    ll_H0 = minus2ll(fit)
    ll_H1 = sum(minus2ll ∘ observed, models.sems)
    return ll_H0 - ll_H1
end
