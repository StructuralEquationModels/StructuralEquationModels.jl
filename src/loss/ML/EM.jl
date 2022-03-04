############################################################################
### Expectation Maximization Algorithm
############################################################################

# An EM Algorithm for MVN-distributed Data with missing values
# Adapted from supplementary Material to the book Machine Learning: A Probabilistic Perspective
# Copyright (2010) Kevin Murphy and Matt Dunham
# found at https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m
# and at https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m

# what about random restarts?

# outer function -----------------------------------------------------------------

function em_mvn(
    ;observed::SemObsMissing,
    start_em = start_em_observed,
    max_iter = 100,
    rtol = 1e-4,
    kwargs...)
    
    n_obs, n_man = observed.n_obs, Int(observed.n_man)

    # preallocate stuff?
    𝔼x_pre = zeros(n_man)
    𝔼xxᵀ_pre = zeros(n_man, n_man)

    ### precompute for full cases
    if length(observed.patterns[1]) == observed.n_man
        for row ∈ observed.rows[1]
            row = observed.data_rowwise[row]
            𝔼x_pre += row;
            𝔼xxᵀ_pre += row*row';
        end
    end
    
    # ess = 𝔼x, 𝔼xxᵀ, ismissing, missingRows, n_obs
    # estepFn = (em_model, data) -> estep(em_model, data, EXsum, EXXsum, ismissing, missingRows, n_obs)

    # initialize
    em_model = start_em(observed; kwargs...)
    em_model_prev = EmMVNModel(zeros(n_man, n_man), zeros(n_man))
    iter = 1
    done = false
    𝔼x = zeros(n_man)
    𝔼xxᵀ = zeros(n_man, n_man)

    while !done

        em_mvn_Estep!(𝔼x, 𝔼xxᵀ, em_model, observed, 𝔼x_pre, 𝔼xxᵀ_pre)
        em_mvn_Mstep!(em_model, n_obs, 𝔼x, 𝔼xxᵀ)

        if iter > max_iter
            done = true
            @warn "EM Algorithm for MVN missing data did not converge. Likelihood for FIML is not interpretable. 
            Maybe try passing different starting values via 'start_em = ...' "
        elseif iter > 1
            # done = isapprox(ll, ll_prev; rtol = rtol)
            done = isapprox(em_model_prev.μ, em_model.μ; rtol = rtol) & isapprox(em_model_prev.Σ, em_model.Σ; rtol = rtol)
        end

        # print("$iter \n")
        iter = iter + 1
        em_model_prev.μ, em_model_prev.Σ = em_model.μ, em_model.Σ

    end

    return iter, em_model

end

# Type to store result -----------------------------------------------------------------

mutable struct EmMVNModel{A, b}
    Σ::A
    μ::b
end

# generate starting values --------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(observed::SemObsMissing, kwargs...)

    if (length(observed.patterns[1]) == observed.n_man) & (observed.pattern_n_obs[1] > 1)
        μ = copy(observed.obs_mean[1])
        Σ = copy(Symmetric(observed.obs_cov[1]))
        if !isposdef(Σ)
            Σ = Matrix(Diagonal(Σ))
        end
        return EmMVNModel(Σ, μ)
    else
        @warn "Could not use Cov and Mean of observed samples as starting values for EM.
        Fall back to simple starting values"
        return start_em_simple(observed, kwargs...)
    end

end

# use μ = O and Σ = I
function start_em_simple(observed::SemObsMissing, kwargs...)
    n_man = Int(observed.n_man)
    μ = zeros(n_man)
    Σ = rand(n_man, n_man); Σ = Σ*Σ'
    # Σ = Matrix(1.0I, n_man, n_man)
    return EmMVNModel(Σ, μ)
end

# set to passed values
function start_em_set(observed::SemObsMissing; model_em, kwargs...)
    return em_model
end

# E and M step ------------------------------------------------------------------------------

function em_mvn_Estep!(𝔼x, 𝔼xxᵀ, em_model, observed, 𝔼x_pre, 𝔼xxᵀ_pre)

    𝔼x .= 0.0
    𝔼xxᵀ .= 0.0

    𝔼xᵢ = copy(𝔼x)
    𝔼xxᵀᵢ = copy(𝔼xxᵀ)

    μ = em_model.μ
    Σ = em_model.Σ

    # Compute the expected sufficient statistics
    for i in 2:length(observed.pattern_n_obs)

        # observed and unobserved vars
        u = observed.patterns_not[i]
        o = observed.patterns[i]

        # precompute for pattern
        V = Σ[u, u] - Σ[u, o] * (Σ[o, o]\Σ[o, u])

        # loop trough data
        for row in observed.rows[i]
            m = μ[u] + Σ[u, o] * ( Σ[o, o] \ (observed.data_rowwise[row]-μ[o]) )

            𝔼xᵢ[u] = m
            𝔼xᵢ[o] = observed.data_rowwise[row]
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
    
function em_mvn_Mstep!(em_model, n_obs, 𝔼x, 𝔼xxᵀ)
    
    em_model.μ = 𝔼x/n_obs;
    Σ = Symmetric(𝔼xxᵀ/n_obs - em_model.μ*em_model.μ')
    
    # ridge Σ
    # while !isposdef(Σ)
    #     Σ += 0.5I
    # end

    em_model.Σ = Σ

    # diagonalization
    #if !isposdef(Σ)
    #    print("Matrix not positive definite")
    #    em_model.Σ .= 0
    #    em_model.Σ[diagind(em_model.Σ)] .= diag(Σ)
    #else
        # em_model.Σ = Σ
    #end

    return nothing
end