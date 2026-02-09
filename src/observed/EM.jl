############################################################################################
### Expectation Maximization Algorithm
############################################################################################

# An EM Algorithm for MVN-distributed Data with missing values
# Adapted from https://github.com/probml/pmtk3, licensed as
#= The MIT License

Copyright (2010) Kevin Murphy and Matt Dunham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. =#

"""
    em_mvn(patterns::AbstractVector{SemObservedMissingPattern};
           max_iter_em = 100,
           rtol_em = 1e-4,
           max_nsamples_em = nothing,
           min_eigval = nothing,
           start_em = start_em_observed,
           start_kwargs...)

Estimate the covariance and the mean for data with missing values using
the expectation maximization (EM) algorithm.

# Arguments
- `patterns`: the observed data with missing values, grouped by missingness pattern (see [`
  SemObservedMissingPattern`](@ref))
- `max_iter_em`: the maximum number of EM iterations
- `rtol_em`: the relative tolerance for convergence of the EM algorithm
- `max_nsamples_em`: the maximum number of samples to use for each pattern in each EM iteration,
  by default all samples are used, but for large datasets it may be desirable to use a random
  subset of the data for each pattern in each EM iteration to speed up the algorithm
- `min_eigval`: the minimum eigenvalue for the covariance matrix;
   if not `nothing`, the covariance matrix is regularized in each EM iteration to ensure that
   all eigenvalues are not smaller than `min_eigval`, which can help with convergence;
- `start_em`: the function to generate starting values for the EM algorithm, by default
  `start_em_observed` which uses the mean and covariance of the full cases if available
- `start_kwargs...`: keyword arguments to pass to the `start_em` function

Returns the tuple of the covariance matrix and the mean vector for the estimated
multivariate normal (MVN) distribution.

# References

Based on the EM algorithm for MVN-distributed data with missing values
adapted from the supplementary material to the book *Machine Learning: A Probabilistic Perspective*,
copyright (2010) Kevin Murphy and Matt Dunham: see
[*gaussMissingFitEm.m*](https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m) and
[*emAlgo.m*](https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m) scripts.
"""
function em_mvn(
    patterns::AbstractVector{<:SemObservedMissingPattern};
    max_iter_em::Integer = 100,
    rtol_em::Number = 1e-4,
    max_nsamples_em::Union{Integer, Nothing} = nothing,
    min_eigval::Union{Number, Nothing} = nothing,
    verbose::Bool = false,
    start_em = start_em_observed,
    start_kwargs...,
)
    nobs_vars = nobserved_vars(patterns[1])

    verbose && @info "Estimating N(μ, Σ) for complete observations..."
    𝔼x_full = zeros(nobs_vars)
    𝔼xxᵀ_full = zeros(nobs_vars, nobs_vars)
    nsamples_full = 0
    for pat in patterns
        if nmissed_vars(pat) == 0
            𝔼x_full .+= sum(pat.data, dims = 2)
            mul!(𝔼xxᵀ_full, pat.data, pat.data', 1, 1)
            nsamples_full += nsamples(pat)
        end
    end
    if nsamples_full == 0
        @warn "No full cases in data"
    end

    verbose && @info "Estimating initial μ and Σ..."
    Σ₀, μ = start_em(patterns; start_kwargs...)
    Σ = convert(Matrix, Σ₀)
    @assert all(isfinite, Σ) all(isfinite, μ)
    Σ_prev, μ_prev = copy(Σ), copy(μ)

    iter = 0
    converged = false
    Δμ_rel = NaN
    ΔΣ_rel = NaN
    progress = Progress(
        max_iter_em,
        dt = 1.0,
        showspeed = true,
        desc = "EM inference of MVN(μ, Σ)",
    )
    while !converged && (iter < max_iter_em)
        em_step!(
            Σ,
            μ,
            Σ_prev,
            μ_prev,
            patterns,
            𝔼xxᵀ_full,
            𝔼x_full,
            nsamples_full;
            max_nsamples_em,
            min_eigval,
        )

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
        next!(progress, step = 1, showvalues = [("ΔΣ/Σ", ΔΣ_rel), ("Δμ/μ", Δμ_rel)])
    end
    finish!(progress)

    if !converged
        @warn "EM inference for MVN missing data did not converge in $iter iterations.\n" *
              "Final tolerances: ΔΣ/Σ=$(ΔΣ_rel), Δμ/μ=$(Δμ_rel).\n" *
              "Likelihood for FIML is not interpretable.\n" *
              "Maybe try passing different starting values via 'start_em = ...' "
    else
        verbose &&
            @info "EM for MVN missing data converged in $iter iterations: ΔΣ/Σ=$(ΔΣ_rel), Δμ/μ=$(Δμ_rel)."
    end

    StatsBase._symmetrize!(Σ)

    return Σ, μ
end

# E and M steps combined
function em_step!(
    Σ::AbstractMatrix,
    μ::AbstractVector,
    Σ₀::AbstractMatrix,
    μ₀::AbstractVector,
    patterns::AbstractVector{<:SemObservedMissingPattern},
    𝔼xxᵀ_full::AbstractMatrix,
    𝔼x_full::AbstractVector,
    nsamples_full::Integer;
    max_nsamples_em::Union{Integer, Nothing} = nothing,
    min_eigval::Union{Number, Nothing} = nothing,
)
    # E step: update 𝔼x and 𝔼xxᵀ
    copy!(μ, 𝔼x_full)
    copy!(Σ, 𝔼xxᵀ_full)
    nsamples_used = nsamples_full
    mul!(Σ, μ₀, μ₀', -nsamples_used, 1)
    axpy!(-nsamples_used, μ₀, μ)

    # Compute the expected sufficient statistics
    for pat in patterns
        (nmissed_vars(pat) == 0) && continue # full cases already accounted for

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.measured_mask

        # compute cholesky to speed-up ldiv!()
        Σ₀oo_chol = cholesky(Symmetric(Σ₀[o, o]))
        Σ₀uo = Σ₀[u, o]
        μ₀u = μ₀[u]
        μ₀o = μ₀[o]

        # get pattern observations
        nsamples_pat =
            !isnothing(max_nsamples_em) ? min(max_nsamples_em, nsamples(pat)) :
            nsamples(pat)
        zo =
            nsamples_pat < nsamples(pat) ?
            pat.data[:, sort!(sample(1:nsamples(pat), nsamples_pat, replace = false))] :
            copy(pat.data)
        zo .-= μ₀o # subtract current mean from observations

        𝔼zo = sum(zo, dims = 2)
        𝔼zu = fill!(similar(μ₀u), 0)

        𝔼zzᵀuo = fill!(similar(Σ₀uo), 0)
        𝔼zzᵀuu = nsamples_pat * Σ₀[u, u]
        mul!(𝔼zzᵀuu, Σ₀uo, Σ₀oo_chol \ Σ₀uo', -nsamples_pat, 1)

        # loop through observations
        yᵢo = similar(μ₀o)
        𝔼zᵢu = similar(μ₀u)
        @inbounds for zᵢo in eachcol(zo)
            ldiv!(yᵢo, Σ₀oo_chol, zᵢo)
            mul!(𝔼zᵢu, Σ₀uo, yᵢo)
            mul!(𝔼zzᵀuu, 𝔼zᵢu, 𝔼zᵢu', 1, 1)
            mul!(𝔼zzᵀuo, 𝔼zᵢu, zᵢo', 1, 1)
            𝔼zu .+= 𝔼zᵢu
        end
        # correct 𝔼zzᵀ by adding back μ₀×𝔼z' + 𝔼z'×μ₀
        mul!(𝔼zzᵀuo, μ₀u, 𝔼zo', 1, 1)
        mul!(𝔼zzᵀuo, 𝔼zu, μ₀o', 1, 1)

        mul!(𝔼zzᵀuu, μ₀u, 𝔼zu', 1, 1)
        mul!(𝔼zzᵀuu, 𝔼zu, μ₀u', 1, 1)

        𝔼zzᵀoo = zo * zo'
        mul!(𝔼zzᵀoo, μ₀o, 𝔼zo', 1, 1)
        mul!(𝔼zzᵀoo, 𝔼zo, μ₀o', 1, 1)

        # update Σ and μ
        Σ[o, o] .+= 𝔼zzᵀoo
        Σ[u, o] .+= 𝔼zzᵀuo
        Σ[o, u] .+= 𝔼zzᵀuo'
        Σ[u, u] .+= 𝔼zzᵀuu

        μ[o] .+= 𝔼zo
        μ[u] .+= 𝔼zu

        nsamples_used += nsamples_pat
    end

    # M step: update Σ and μ
    lmul!(1 / nsamples_used, Σ)
    lmul!(1 / nsamples_used, μ)
    # at this point μ = μ - μ₀
    # and Σ = Σ + (μ - μ₀)×(μ - μ₀)' - μ₀×μ₀'
    mul!(Σ, μ, μ₀', -1, 1)
    mul!(Σ, μ₀, μ', -1, 1)
    mul!(Σ, μ, μ', -1, 1)
    μ .+= μ₀

    # try to fix non-positive-definite Σ
    isnothing(min_eigval) || copyto!(Σ, trunc_eigvals(Σ, min_eigval))

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
