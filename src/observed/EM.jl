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
    nvars = nobserved_vars(observed)
    nsamps = nsamples(observed)

    # preallocate stuff
    𝔼x_pre = zeros(nvars)
    𝔼xxᵀ_pre = zeros(nvars, nvars)

    ### precompute for full cases
    fullpat = observed.patterns[1]
    if nmissed_vars(fullpat) == 0
        for row in eachrow(fullpat.data)
            𝔼x_pre += row
            𝔼xxᵀ_pre += row * row'
        end
    end

    # initialize
    em_model = start_em(observed; kwargs...)
    em_model_prev = EmMVNModel(zeros(nvars, nvars), zeros(nvars), false)
    iter = 1
    done = false
    𝔼x = zeros(nvars)
    𝔼xxᵀ = zeros(nvars, nvars)

    while !done
        em_mvn_Estep!(𝔼x, 𝔼xxᵀ, em_model, observed, 𝔼x_pre, 𝔼xxᵀ_pre)
        em_mvn_Mstep!(em_model, nsamps, 𝔼x, 𝔼xxᵀ)

        if iter > max_iter_em
            done = true
            @warn "EM Algorithm for MVN missing data did not converge. Likelihood for FIML is not interpretable.
            Maybe try passing different starting values via 'start_em = ...' "
        elseif iter > 1
            done =
                isapprox(em_model_prev.μ, em_model.μ; rtol = rtol_em) &
                isapprox(em_model_prev.Σ, em_model.Σ; rtol = rtol_em)
        end

        iter = iter + 1
        em_model_prev.μ, em_model_prev.Σ = em_model.μ, em_model.Σ
    end

    # update EM Mode in observed
    observed.em_model.Σ .= em_model.Σ
    observed.em_model.μ .= em_model.μ
    observed.em_model.fitted = true

    return nothing
end

# E and M step -----------------------------------------------------------------------------

function em_mvn_Estep!(𝔼x, 𝔼xxᵀ, em_model, observed, 𝔼x_pre, 𝔼xxᵀ_pre)
    𝔼x .= 0.0
    𝔼xxᵀ .= 0.0

    𝔼xᵢ = copy(𝔼x)
    𝔼xxᵀᵢ = copy(𝔼xxᵀ)

    μ = em_model.μ
    Σ = em_model.Σ

    # Compute the expected sufficient statistics
    for pat in observed.patterns
        (nmissed_vars(pat) == 0) && continue # skip full cases

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.measured_mask

        # precompute for pattern
        Σoo = Σ[o, o]
        Σuo = Σ[u, o]
        μu = μ[u]
        μo = μ[o]

        V = Σ[u, u] - Σuo * (Σoo \ Σ[o, u])

        # loop trough data
        for rowdata in eachrow(pat.data)
            m = μu + Σuo * (Σoo \ (rowdata - μo))

            𝔼xᵢ[u] = m
            𝔼xᵢ[o] = rowdata
            𝔼xxᵀᵢ[u, u] = 𝔼xᵢ[u] * 𝔼xᵢ[u]' + V
            𝔼xxᵀᵢ[o, o] = 𝔼xᵢ[o] * 𝔼xᵢ[o]'
            𝔼xxᵀᵢ[o, u] = 𝔼xᵢ[o] * 𝔼xᵢ[u]'
            𝔼xxᵀᵢ[u, o] = 𝔼xᵢ[u] * 𝔼xᵢ[o]'

            𝔼x .+= 𝔼xᵢ
            𝔼xxᵀ .+= 𝔼xxᵀᵢ
        end
    end

    𝔼x .+= 𝔼x_pre
    𝔼xxᵀ .+= 𝔼xxᵀ_pre
end

function em_mvn_Mstep!(em_model, nsamples, 𝔼x, 𝔼xxᵀ)
    em_model.μ = 𝔼x / nsamples
    Σ = Symmetric(𝔼xxᵀ / nsamples - em_model.μ * em_model.μ')
    em_model.Σ = Σ
    return nothing
end

# generate starting values -----------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(observed::SemObservedMissing; kwargs...)
    fullpat = observed.patterns[1]
    if (nmissed_vars(fullpat) == 0) && (nobserved_vars(fullpat) > 1)
        μ = copy(fullpat.measured_mean)
        Σ = copy(Symmetric(fullpat.measured_cov))
        if !isposdef(Σ)
            Σ = Matrix(Diagonal(Σ))
        end
        return EmMVNModel(Σ, μ, false)
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
