"""
    se_hessian(fit::SemFit; method = :finitediff)

Return hessian-based standard errors.

# Arguments
- `method`: how to compute the hessian. Options are
    - `:analytic`: (only if an analytic hessian for the model can be computed)
    - `:finitediff`: for finite difference approximation
"""
function se_hessian(fit::SemFit; method = :finitediff)

    c = H_scaling(fit.model)
    params = solution(fit)
    H = similar(params, (length(params), length(params)))

    if method == :analytic
        evaluate!(nothing, nothing, H, fit.model, params)
    elseif method == :finitediff
        FiniteDiff.finite_difference_hessian!(H,
            p -> evaluate!(eltype(H), nothing, nothing, fit.model, p), params)
    elseif method == :optimizer
        error("Standard errors from the optimizer hessian are not implemented yet")
    elseif method == :expected
        error("Standard errors based on the expected hessian are not implemented yet")
    else
        throw(ArgumentError("Unsupported hessian calculation method :$method"))
    end

    H_chol = cholesky!(Symmetric(H))
    H_inv = LinearAlgebra.inv!(H_chol)
    return [sqrt(c*H_inv[i]) for i in diagind(H_inv)]
end

# Addition functions -------------------------------------------------------------
H_scaling(model::AbstractSemSingle) =
    H_scaling(
        model,
        model.observed,
        model.imply,
        model.optimizer,
        model.loss.functions...)

H_scaling(model, obs, imp, optimizer, lossfun::SemML) =
    2/(n_obs(model)-1)

function H_scaling(model, obs, imp, optimizer, lossfun::SemWLS)
    @warn "Standard errors for WLS are only correct if a GLS weight matrix (the default) is used."
    return 2/(n_obs(model)-1)
end

H_scaling(model, obs, imp, optimizer, lossfun::SemFIML) =
    2/n_obs(model)

H_scaling(model::SemEnsemble) =
    2/n_obs(model)