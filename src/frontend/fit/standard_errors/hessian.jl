############################################################################
### observed
############################################################################

function se_hessian(sem_fit::SemFit; hessian = :finitediff)

    c = H_scaling(sem_fit.model)

    if hessian == :analytic
        H = hessian!(sem_fit.model, sem_fit.solution)
    elseif hessian == :analytic_last
        H = hessian(sem_fit.model)
    elseif hessian == :finitediff
        H = FiniteDiff.finite_difference_hessian(
                x -> objective!(sem_fit.model, x), 
                sem_fit.solution
                )
    elseif hessian == :optimizer
        @error "Standard errors from the optimizer hessian are not implemented yet"
    elseif hessian == :expected
        @error "Standard errors based on the expected hessian are not implemented yet"
    else
        @error "I dont know how to compute $how standard-errors"
    end

    invH = c*inv(H)
    se = sqrt.(diag(invH))

    return se
end

# Addition functions -------------------------------------------------------------
H_scaling(model::AbstractSemSingle) = 
    H_scaling(
        model,
        model.observed,
        model.imply,
        model.diff,
        model.loss.functions...)

H_scaling(model, obs, imp, diff, lossfun::Union{SemML, SemWLS}) =
    2/(n_obs(model)-1)

H_scaling(model, obs, imp, diff, lossfun::SemFIML) =
    2/(n_obs(model))

H_scaling(model::SemEnsemble) =
    2/(n_obs(model)-1)

############################################################################
### expected
############################################################################

# not available yet