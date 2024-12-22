############################################################################################
### Expectation Maximization Algorithm
############################################################################################

# what about random restarts?

"""
    em_mvn(patterns::AbstractVector{SemObservedMissingPattern};
           start_em = start_em_observed,
           max_iter_em = 100,
           rtol_em = 1e-4,
           kwargs...)

Estimates the covariance matrix and mean vector of the
multivariate normal distribution (MVN)
via expectation maximization (EM) for `observed`.

Returns the tuple of the EM covariance matrix and the EM mean vector.

Uses the EM algorithm for MVN-distributed data with missing values
adapted from the supplementary material to the book *Machine Learning: A Probabilistic Perspective*,
copyright (2010) Kevin Murphy and Matt Dunham: see
[*gaussMissingFitEm.m*](https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m) and
[*emAlgo.m*](https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m) scripts.
"""
function em_mvn(
    patterns::AbstractVector{<:SemObservedMissingPattern};
    start_em = start_em_observed,
    max_iter_em::Integer = 100,
    rtol_em::Number = 1e-4,
    kwargs...,
)
    nobs_vars = nobserved_vars(patterns[1])

    ### precompute for full cases
    𝔼x_full = zeros(nobs_vars)
    𝔼xxᵀ_full = zeros(nobs_vars, nobs_vars)
    if nmissed_vars(patterns[1]) == 0
        fullpat = patterns[1]
        sum!(reshape(𝔼x_full, 1, nobs_vars), fullpat.data)
        mul!(𝔼xxᵀ_full, fullpat.data', fullpat.data)
    else
        @warn "No full cases pattern found"
    end

    # ess = 𝔼x, 𝔼xxᵀ, ismissing, missingRows, nsamps
    # estepFn = (em_model, data) -> estep(em_model, data, EXsum, EXXsum, ismissing, missingRows, nsamps)

    # initialize
    Σ₀, μ = start_em(patterns; kwargs...)
    Σ = convert(Matrix, Σ₀)
    @assert all(isfinite, Σ) all(isfinite, μ)
    Σ_prev, μ_prev = copy(Σ), copy(μ)

    iter = 0
    converged = false
    while !converged && (iter < max_iter_em)
        em_step!(Σ, μ, Σ_prev, μ_prev, patterns, 𝔼x_full, 𝔼xxᵀ_full)

        if iter > 0
            Δμ = norm(μ - μ_prev)
            ΔΣ = norm(Σ - Σ_prev)
            Δμ_rel = Δμ / max(norm(μ_prev), norm(μ))
            ΔΣ_rel = ΔΣ / max(norm(Σ_prev), norm(Σ))
            #@info "Iteration #$iter: ΔΣ=$(ΔΣ) ΔΣ/Σ=$(ΔΣ_rel) Δμ=$(Δμ) Δμ/μ=$(Δμ_rel)"
            # converged = isapprox(ll, ll_prev; rtol = rtol)
            converged = ΔΣ_rel <= rtol_em && Δμ_rel <= rtol_em
        end
        if !converged
            Σ, Σ_prev = Σ_prev, Σ
            μ, μ_prev = μ_prev, μ
        end
        iter += 1
        #@info "$iter\n"
    end

    if !converged
        @warn "EM Algorithm for MVN missing data did not converge in $iter iterations.\n" *
              "Likelihood for FIML is not interpretable.\n" *
              "Maybe try passing different starting values via 'start_em = ...' "
    else
        @info "EM for MVN missing data converged in $iter iterations"
    end

    return Σ, μ
end

# E and M steps -----------------------------------------------------------------------------

function em_step!(
    Σ::AbstractMatrix,
    μ::AbstractVector,
    Σ₀::AbstractMatrix,
    μ₀::AbstractVector,
    patterns::AbstractVector{<:SemObservedMissingPattern},
    𝔼x_full,
    𝔼xxᵀ_full,
)
    # E step, update 𝔼x and 𝔼xxᵀ
    copy!(μ, 𝔼x_full)
    copy!(Σ, 𝔼xxᵀ_full)

    # Compute the expected sufficient statistics
    for pat in patterns
        (nmissed_vars(pat) == 0) && continue # skip full cases

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.measured_mask

        # precompute for pattern
        Σoo_chol = cholesky(Symmetric(Σ₀[o, o]))
        Σuo = Σ₀[u, o]
        μu = μ₀[u]
        μo = μ₀[o]

        𝔼xu = fill!(similar(μu), 0)
        𝔼xo = fill!(similar(μo), 0)
        𝔼xᵢu = similar(μu)

        𝔼xxᵀuo = fill!(similar(Σuo), 0)
        𝔼xxᵀuu = n_obs(pat) * (Σ₀[u, u] - Σuo * (Σoo_chol \ Σuo'))

        # loop trough data
        @inbounds for rowdata in eachrow(pat.data)
            mul!(𝔼xᵢu, Σuo, Σoo_chol \ (rowdata - μo))
            𝔼xᵢu .+= μu
            mul!(𝔼xxᵀuu, 𝔼xᵢu, 𝔼xᵢu', 1, 1)
            mul!(𝔼xxᵀuo, 𝔼xᵢu, rowdata', 1, 1)
            𝔼xu .+= 𝔼xᵢu
            𝔼xo .+= rowdata
        end

        Σ[o, o] .+= pat.data' * pat.data
        Σ[u, o] .+= 𝔼xxᵀuo
        Σ[o, u] .+= 𝔼xxᵀuo'
        Σ[u, u] .+= 𝔼xxᵀuu

        μ[o] .+= 𝔼xo
        μ[u] .+= 𝔼xu
    end

    # M step, update em_model
    k = inv(sum(nsamples, patterns))
    lmul!(k, Σ)
    lmul!(k, μ)
    mul!(Σ, μ, μ', -1, 1)

    # ridge Σ
    # while !isposdef(Σ)
    #     Σ += 0.5I
    # end

    # diagonalization
    #if !isposdef(Σ)
    #    print("Matrix not positive definite")
    #    Σ .= 0
    #    Σ[diagind(em_model.Σ)] .= diag(Σ)
    #else
    # Σ = Σ
    #end

    return Σ, μ
end

# generate starting values -----------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(patterns::AbstractVector{<:SemObservedMissingPattern}; kwargs...)
    fullpat = patterns[1]
    if (nmissed_vars(fullpat) == 0) && (nsamples(fullpat) > 1)
        μ = copy(fullpat.measured_mean)
        Σ = copy(parent(fullpat.measured_cov))
        if !isposdef(Σ)
            Σ = Diagonal(Σ)
        end
        return Σ, μ
    else
        return start_em_simple(patterns, kwargs...)
    end
end

# use μ = O and Σ = I
function start_em_simple(patterns::AbstractVector{<:SemObservedMissingPattern}; kwargs...)
    nobs_vars = nobserved_vars(first(patterns))
    μ = zeros(nobs_vars)
    Σ = rand(nobs_vars, nobs_vars)
    Σ = Σ * Σ'
    # Σ = Matrix(1.0I, nobs_vars, nobs_vars)
    return Σ, μ
end

# set to passed values
function start_em_set(
    patterns::AbstractVector{<:SemObservedMissingPattern};
    obs_cov::AbstractMatrix,
    obs_mean::AbstractVector,
    kwargs...,
)
    return copy(obs_cov), copy(obs_mean)
end
