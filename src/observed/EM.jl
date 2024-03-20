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
    em_mvn!(;
        observed::SemObservedMissing,
        start_em = start_em_observed,
        max_iter_em = 100,
        rtol_em = 1e-4,
        kwargs...)

Estimates the covariance matrix and mean vector of the normal distribution via expectation maximization for `observed`.
Overwrites the statistics stored in `observed`.

Uses the EM algorithm for MVN-distributed data with missing values
adapted from the supplementary material to the book *Machine Learning: A Probabilistic Perspective*,
copyright (2010) Kevin Murphy and Matt Dunham: see
[*gaussMissingFitEm.m*](https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m) and
[*emAlgo.m*](https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m) scripts.
"""
function em_mvn!(
    observed::SemObservedMissing;
    start_em = start_em_observed,
    max_iter_em = 100,
    rtol_em = 1e-4,
    kwargs...,
)
    nobs_vars = nobserved_vars(observed)

    # precompute for full cases
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

    # initialize
    em_model = start_em(observed; kwargs...)
    em_model_prev = EmMVNModel(zeros(nobs_vars, nobs_vars), zeros(nobs_vars), false)
    iter = 1
    done = false
    𝔼x = zeros(nobs_vars)
    𝔼xxᵀ = zeros(nobs_vars, nobs_vars)

    while !done
        step!(em_model, observed, 𝔼x, 𝔼xxᵀ, 𝔼x_pre, 𝔼xxᵀ_pre)

        if iter > max_iter_em
            done = true
            @warn "EM Algorithm for MVN missing data did not converge. Likelihood for FIML is not interpretable.
            Maybe try passing different starting values via 'start_em = ...' "
        elseif iter > 1
            done =
                isapprox(em_model_prev.μ, em_model.μ; rtol = rtol_em) &&
                isapprox(em_model_prev.Σ, em_model.Σ; rtol = rtol_em)
        end

        # print("$iter \n")
        iter += 1
        copyto!(em_model_prev.μ, em_model.μ)
        copyto!(em_model_prev.Σ, em_model.Σ)
    end

    # update EM Mode in observed
    observed.em_model.Σ .= em_model.Σ
    observed.em_model.μ .= em_model.μ
    observed.em_model.fitted = true

    return nothing
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
    for pat in observed.patterns
        (nmissed_vars(pat) == 0) && continue # skip full cases

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.measured_mask

        # precompute for pattern
        Σoo_chol = cholesky(Symmetric(Σ[o, o]))
        Σuo = Σ[u, o]
        μu = μ[u]
        μo = μ[o]

        𝔼xu = fill!(similar(μu), 0)
        𝔼xo = fill!(similar(μo), 0)
        𝔼xᵢu = similar(μu)

        𝔼xxᵀuo = fill!(similar(Σuo), 0)
        𝔼xxᵀuu = n_obs(pat) * (Σ[u, u] - Σuo * (Σoo_chol \ Σuo'))

        # loop through observations
        @inbounds for rowdata in eachcol(pat.data)
            mul!(𝔼xᵢu, Σuo, Σoo_chol \ (rowdata - μo))
            𝔼xᵢu .+= μu
            mul!(𝔼xxᵀuu, 𝔼xᵢu, 𝔼xᵢu', 1, 1)
            mul!(𝔼xxᵀuo, 𝔼xᵢu, rowdata', 1, 1)
            𝔼xu .+= 𝔼xᵢu
            𝔼xo .+= rowdata
        end

        𝔼xxᵀ[o, o] .+= pat.data' * pat.data
        𝔼xxᵀ[u, o] .+= 𝔼xxᵀuo
        𝔼xxᵀ[o, u] .+= 𝔼xxᵀuo'
        𝔼xxᵀ[u, u] .+= 𝔼xxᵀuu

        𝔼x[o] .+= 𝔼xo
        𝔼x[u] .+= 𝔼xu
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

    em_model.μ .= 𝔼x ./ nsamples(observed)
    em_model.Σ .= 𝔼xxᵀ ./ nsamples(observed)
    mul!(em_model.Σ, em_model.μ, em_model.μ', -1, 1)

    return em_model
end

# generate starting values -----------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(observed::SemObservedMissing; kwargs...)
    fullpat = observed.patterns[1]
    if (nmissed_vars(fullpat) == 0) && (nsamples(fullpat) > 1)
        μ = copy(fullpat.measured_mean)
        Σ = copy(fullpat.measured_cov)
        if !isposdef(Σ)
            Σ = Diagonal(Σ)
        end
        return EmMVNModel(convert(Matrix, Σ), μ, false)
    else
        return start_em_simple(observed, kwargs...)
    end
end

# use μ = O and Σ = I
function start_em_simple(observed::SemObservedMissing; kwargs...)
    nvars = nobserved_vars(observed)
    μ = zeros(nvars)
    Σ = rand(nvars, nvars)
    Σ = Σ * Σ'
    return EmMVNModel(Σ, μ, false)
end

# set to passed values
function start_em_set(observed::SemObservedMissing; model_em, kwargs...)
    return em_model
end
