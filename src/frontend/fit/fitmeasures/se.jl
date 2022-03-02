# SemFit splices loss functions ---------------------------------------------------------------------
p_value(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = 
    p_value(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
p_value(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemML) = p_value(sem_fit.minimum, obs, imp)
p_value(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemML) = p_value(sem_fit.minimum, obs, imp)

function p_value(minimum, observed::SemObsCommon, imply)
    chi2 = χ²(minimum, observed)
    dist_chi2 = Chisq(df(observed, imply))
    return 1 - cdf(dist_chi2, chi2)
end

function se_hessian(sem_fit::SemFit; analytic = false)
    if analytic
        hessian = hessian!(sem_fit.model, sem_fit.solution)
    else
        hessian = 
            FiniteDiff.finite_difference_hessian(
                x -> objective!(sem_fit.model, x), 
                sem_fit.solution)
    end

    invH = inv(0.5(sem_fit.model.observed.n_obs-1)*hessian)
    se = sqrt.(diag(invH))
    return se
end

#####

function se_fisher(sem_fit::SemFit; analytic = false)
    gradient = 0.5(sem_fit.model.observed.n_obs-1)*gradient!(sem_fit.model, sem_fit.solution)
    hessian = gradient*gradient'
    # return hessian
    invH = hessian
    se = sqrt.(diag(invH))
    return se
end