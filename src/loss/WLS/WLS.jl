##### weighted least squares

############################################################################################
### Types
############################################################################################
"""
Weighted least squares estimation.
At the moment only available with the `RAMSymbolic` implied type.

# Constructor

    SemWLS(
        observed::SemObserved, implied::SemImplied;
        wls_weight_matrix = nothing,
        wls_weight_matrix_mean = nothing,
        approximate_hessian = false,
        kwargs...)

# Arguments
- `observed`: the `SemObserved` part of the model
- `implied`: the `SemImplied` part of the model
- `approximate_hessian::Bool`: should the hessian be swapped for an approximation
- `wls_weight_matrix`: the weight matrix for weighted least squares.
    Defaults to GLS estimation (``0.5*(D^T*kron(S,S)*D)`` where D is the duplication matrix
    and S is the inverse of the observed covariance matrix)
- `wls_weight_matrix_mean`: the weight matrix for the mean part of weighted least squares.
    Defaults to GLS estimation (the inverse of the observed covariance matrix)

# Examples
```julia
my_wls = SemWLS(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure also analytic hessians.
"""
struct SemWLS{O, I, HE <: HessianEval, Vt, St, C} <: SemLoss{O, I}
    observed::O
    implied::I

    hessianeval::HE
    V::Vt
    σₒ::St
    V_μ::C

    SemWLS(observed, implied, ::Type{HE}, args...) where {HE <: HessianEval} =
        new{typeof(observed), typeof(implied), HE, map(typeof, args)...}(
            observed,
            implied,
            HE(),
            args...,
        )
end

############################################################################################
### Constructors
############################################################################################

function SemWLS(
    observed::SemObserved,
    implied::SemImplied;
    wls_weight_matrix::Union{AbstractMatrix, Nothing} = nothing,
    wls_weight_matrix_mean::Union{AbstractMatrix, Nothing} = nothing,
    approximate_hessian::Bool = false,
    kwargs...,
)
    if observed isa SemObservedMissing
        @warn """
        WLS estimation with `SemObservedMissing` will use an approximate covariance and mean estimated with EM algorithm.
        For more accurate results, consider using full information maximum likelihood (FIML) estimation or remove missing values in your data.
        A FIML model can be constructed with
        Sem(
            ...,
            observed = SemObservedMissing,
            loss = SemFIML,
            meanstructure = true
        )
        """
    end

    if !(implied isa RAMSymbolic)
        "WLS estimation is only available with the implied type RAMSymbolic at the moment." |>
        ArgumentError |>
        throw
    end
    # check integrity
    check_observed_vars(observed, implied)

    nobs_vars = nobserved_vars(observed)
    tril_ind = filter(x -> (x[1] >= x[2]), CartesianIndices(obs_cov(observed)))
    s = obs_cov(observed)[tril_ind]
    size(s) == size(implied.Σ) || throw(
        DimensionMismatch(
            "SemWLS requires implied covariance to be in vech-ed form " *
            "(vectorized lower triangular part of Σ matrix): $(size(s)) expected, $(size(implied.Σ)) found.\n" *
            "$(nameof(typeof(implied))) must be constructed with vech=true.",
        ),
    )

    # compute V here
    if isnothing(wls_weight_matrix)
        D = duplication_matrix(nobs_vars)
        S = inv(obs_cov(observed))
        S = kron(S, S)
        wls_weight_matrix = 0.5 * (D' * S * D)
    else
        size(wls_weight_matrix) == (length(tril_ind), length(tril_ind)) ||
            DimensionMismatch(
                "wls_weight_matrix has to be of size $(length(tril_ind))×$(length(tril_ind))",
            )
    end
    size(wls_weight_matrix) == (length(s), length(s)) ||
        DimensionMismatch("wls_weight_matrix has to be of size $(length(s))×$(length(s))")

    if MeanStruct(implied) == HasMeanStruct
        if isnothing(wls_weight_matrix_mean)
            @info "Computing WLS weight matrix for the meanstructure using obs_cov()"
            wls_weight_matrix_mean = inv(obs_cov(observed))
        end
        size(wls_weight_matrix_mean) == (nobs_vars, nobs_vars) || DimensionMismatch(
            "wls_weight_matrix_mean has to be of size $(nobs_vars)×$(nobs_vars)",
        )
    else
        isnothing(wls_weight_matrix_mean) ||
            @warn "Ignoring wls_weight_matrix_mean since meanstructure is disabled"
        wls_weight_matrix_mean = nothing
    end
    HE = approximate_hessian ? ApproxHessian : ExactHessian

    return SemWLS(observed, implied, HE, wls_weight_matrix, s, wls_weight_matrix_mean)
end

############################################################################
### methods
############################################################################

function evaluate!(objective, gradient, hessian, loss::SemWLS, par)
    implied = SEM.implied(loss)

    if !isnothing(hessian) && (MeanStruct(implied) === HasMeanStruct)
        error("hessian of WLS with meanstructure is not available")
    end

    V = loss.V
    ∇σ = implied.∇Σ

    σ = implied.Σ
    σₒ = loss.σₒ
    σ₋ = σₒ - σ

    isnothing(objective) || (objective = dot(σ₋, V, σ₋))
    if !isnothing(gradient)
        if issparse(∇σ)
            gradient .= (σ₋' * V * ∇σ)'
        else # save one allocation
            mul!(gradient, σ₋' * V, ∇σ)
        end
        gradient .*= -2
    end
    isnothing(hessian) || (mul!(hessian, ∇σ' * V, ∇σ, 2, 0))
    if !isnothing(hessian) && (HessianEval(loss) === ExactHessian)
        ∇²Σ = implied.∇²Σ
        J = -2 * (σ₋' * loss.V)'
        implied.∇²Σ_eval!(∇²Σ, J, par)
        hessian .+= ∇²Σ
    end
    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(loss))
        μ₋ = μₒ - μ
        V_μ = loss.V_μ
        if !isnothing(objective)
            objective += dot(μ₋, V_μ, μ₋)
        end
        if !isnothing(gradient)
            mul!(gradient, (V_μ * implied.∇μ)', μ₋, -2, 1)
        end
    end

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

function update_observed(
    loss::SemWLS,
    observed::SemObserved;
    recompute_V = true,
    kwargs...,
)
    if recompute_V
        return SemWLS(observed, loss.implied; kwargs...)
    else
        return SemWLS(
            observed,
            loss.implied;
            wls_weight_matrix = loss.V,
            wls_weight_matrix_mean = loss.V_μ,
            kwargs...,
        )
    end
end
