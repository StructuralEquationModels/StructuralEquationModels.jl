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
        FiniteDiff.finite_difference_hessian!(
            H,
            p -> evaluate!(zero(eltype(H)), nothing, nothing, fit.model, p),
            params,
        )
    elseif method == :optimizer
        error("Standard errors from the optimizer hessian are not implemented yet")
    elseif method == :expected
        error("Standard errors based on the expected hessian are not implemented yet")
    else
        throw(ArgumentError("Unsupported hessian calculation method :$method"))
    end

    H_chol = cholesky!(Symmetric(H))
    H_inv = LinearAlgebra.inv!(H_chol)
    return [sqrt(c * H_inv[i]) for i in diagind(H_inv)]
end

# Additional functions -------------------------------------------------------------

H_nsamples(loss::SemML) = nsamples(loss) - 1
H_nsamples(loss::SemWLS) = nsamples(loss) - 1
H_nsamples(loss::SemFIML) = nsamples(loss)
H_nsamples(loss::SemLoss) = nsamples(loss)
H_nsamples(wrapper::SemLossFiniteDiff) = H_nsamples(_unwrap(wrapper))

function H_scaling(model::AbstractSem)
    semterms = SEM.sem_terms(model)
    isempty(semterms) && return 1.0
    if any(term -> _unwrap(loss(term)) isa SemWLS, semterms)
        @warn "Standard errors for WLS are only correct if a GLS weight matrix (the default) is used."
    end
    return 2 / sum(term -> H_nsamples(loss(term)), semterms)
end
