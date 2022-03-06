############################################################################
### observed
############################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
se_hessian(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}; analytic = false) = 
    se_hessian(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...;
        analytic = analytic
        )

# RAM + SemML
se_hessian(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemML; analytic) = se_hessian(sem_fit; analytic = analytic)
se_hessian(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemML; analytic) = se_hessian(sem_fit; analytic = analytic)

# se_hessian(sem_fit::SemFit, obs, imp::RAM, diff, loss_ml::SemFIML; analytic) = se_hessian(sem_fit; analytic = analytic)
# se_hessian(sem_fit::SemFit, obs, imp::RAMSymbolic, diff, loss_ml::SemFIML; analytic) = se_hessian(sem_fit; analytic = analytic)

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

############################################################################
### expected
############################################################################

# not available yet