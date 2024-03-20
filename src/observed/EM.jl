############################################################################################
### Expectation Maximization Algorithm
############################################################################################

# An EM Algorithm for MVN-distributed Data with missing values
# Adapted from supplementary Material to the book Machine Learning: A Probabilistic Perspective
# Copyright (2010) Kevin Murphy and Matt Dunham
# found at https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m
# and at https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m

# what about random restarts?

# outer function ---------------------------------------------------------------------------
"""
    em_mvn(;
        observed::SemObservedMissing,
        start_em = start_em_observed,
        max_iter_em = 100,
        rtol_em = 1e-4,
        kwargs...)

Estimates the covariance matrix and mean vector of the normal distribution via expectation maximization for `observed`.
Overwrites the statistics stored in `observed`.
"""
function em_mvn(
    observed::SemObservedMissing;
    start_em = start_em_observed,
    max_iter_em = 100,
    rtol_em = 1e-4,
    kwargs...)

    n_man = SEM.n_man(observed)

    # preallocate stuff?
    𝔼x_pre = zeros(n_man)
    𝔼xxᵀ_pre = zeros(n_man, n_man)

    ### precompute for full cases
    fullpat = observed.patterns[1]
    if nmissed_vars(fullpat) == 0
        sum!(reshape(𝔼x_pre, 1, n_man), fullpat.data)
        mul!(𝔼xxᵀ_pre, fullpat.data', fullpat.data)
    else
        @warn "No full cases pattern found"
    end

    # ess = 𝔼x, 𝔼xxᵀ, ismissing, missingRows, n_obs
    # estepFn = (em_model, data) -> estep(em_model, data, EXsum, EXXsum, ismissing, missingRows, n_obs)

    # initialize
    em_model = start_em(observed; kwargs...)
        em_model_prev = EmMVNModel(zeros(n_man, n_man), zeros(n_man), false)
    iter = 1
    done = false
    𝔼x = zeros(n_man)
    𝔼xxᵀ = zeros(n_man, n_man)

    while !done

        step!(em_model, observed, 𝔼x, 𝔼xxᵀ, 𝔼x_pre, 𝔼xxᵀ_pre)

        if iter > max_iter_em
            done = true
            @warn "EM Algorithm for MVN missing data did not converge. Likelihood for FIML is not interpretable.
            Maybe try passing different starting values via 'start_em = ...' "
        elseif iter > 1
            # done = isapprox(ll, ll_prev; rtol = rtol)
            done = isapprox(em_model_prev.μ, em_model.μ; rtol = rtol_em) &&
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

# E and M steps -----------------------------------------------------------------------------

# update em_model
function step!(em_model::EmMVNModel, observed::SemObserved,
               𝔼x, 𝔼xxᵀ, 𝔼x_pre, 𝔼xxᵀ_pre)
    # E step, update 𝔼x and 𝔼xxᵀ
    fill!(𝔼x, 0)
    fill!(𝔼xxᵀ, 0)

    μ = em_model.μ
    Σ = em_model.Σ

    # Compute the expected sufficient statistics
    for pat in observed.patterns
        (nmissed_vars(pat) == 0) && continue # skip full cases

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.obs_mask

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

        # loop trough data
        @inbounds for rowdata in eachrow(pat.data)
            mul!(𝔼xᵢu, Σuo, Σoo_chol \ (rowdata-μo))
            𝔼xᵢu .+= μu
            mul!(𝔼xxᵀuu, 𝔼xᵢu, 𝔼xᵢu', 1, 1)
            mul!(𝔼xxᵀuo, 𝔼xᵢu, rowdata', 1, 1)
            𝔼xu .+= 𝔼xᵢu
            𝔼xo .+= rowdata
        end

        𝔼xxᵀ[o,o] .+= pat.data' * pat.data
        𝔼xxᵀ[u,o] .+= 𝔼xxᵀuo
        𝔼xxᵀ[o,u] .+= 𝔼xxᵀuo'
        𝔼xxᵀ[u,u] .+= 𝔼xxᵀuu

        𝔼x[o] .+= 𝔼xo
        𝔼x[u] .+= 𝔼xu
    end

    𝔼x .+= 𝔼x_pre
    𝔼xxᵀ .+= 𝔼xxᵀ_pre

    # M step, update em_model
    em_model.μ .= 𝔼x ./ n_obs(observed)
    em_model.Σ .= 𝔼xxᵀ ./ n_obs(observed)
    mul!(em_model.Σ, em_model.μ, em_model.μ', -1, 1)

    #Σ = em_model.Σ
    # ridge Σ
    # while !isposdef(Σ)
    #     Σ += 0.5I
    # end

    # diagonalization
    #if !isposdef(Σ)
    #    print("Matrix not positive definite")
    #    em_model.Σ .= 0
    #    em_model.Σ[diagind(em_model.Σ)] .= diag(Σ)
    #else
        # em_model.Σ = Σ
    #end

    return em_model
end

# generate starting values -----------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(observed::SemObservedMissing; kwargs...)

    fullpat = observed.patterns[1]
    if (nmissed_vars(fullpat) == 0) && (n_obs(fullpat) > 1)
        μ = copy(fullpat.obs_mean)
        Σ = copy(fullpat.obs_cov)
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
    μ = zeros(n_man(observed))
    Σ = rand(n_man(observed), n_man(observed))
    Σ = Σ*Σ'
    # Σ = Matrix(1.0I, n_man, n_man)
    return EmMVNModel(Σ, μ, false)
end

# set to passed values
function start_em_set(observed::SemObservedMissing; model_em, kwargs...)
    return em_model
end