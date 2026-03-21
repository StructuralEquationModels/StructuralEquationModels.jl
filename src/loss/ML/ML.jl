# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(observed, implied; approximate_hessian = false)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `implied::SemImplied`: [`SemImplied`](@ref) instance
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure
and `RAMSymbolic` implied type, also analytic hessians.
"""
struct SemML{O, I, HE <: HessianEval, INV, M, M2} <: SemLoss{O, I}
    observed::O
    implied::I
    hessianeval::HE
    Σ⁻¹::INV
    Σ⁻¹Σₒ::M
    meandiff::M2
end

############################################################################################
### Constructors
############################################################################################

function SemML(
    observed::SemObserved,
    implied::SemImplied;
    approximate_hessian::Bool = false,
    kwargs...,
)
    if observed isa SemObservedMissing
        @warn """
        ML estimation with `SemObservedMissing` will use an approximate covariance and mean estimated with EM algorithm.
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
    # check integrity
    check_observed_vars(observed, implied)

    he = approximate_hessian ? ApproxHessian() : ExactHessian()
    obsmean = obs_mean(observed)
    obscov = obs_cov(observed)
    meandiff = isnothing(obsmean) ? nothing : copy(obsmean)

    return SemML{
        typeof(observed),
        typeof(implied),
        typeof(he),
        typeof(obscov),
        typeof(obscov),
        typeof(meandiff),
    }(
        observed,
        implied,
        he,
        similar(obscov),
        similar(obscov),
        meandiff,
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Implied Types

function evaluate!(
    objective,
    gradient,
    hessian,
    loss::SemML{<:Any, <:SemImpliedSymbolic},
    par,
)
    implied = SEM.implied(loss)

    if !isnothing(hessian)
        (MeanStruct(implied) === HasMeanStruct) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(loss))
    Σ⁻¹Σₒ = loss.Σ⁻¹Σₒ
    Σ⁻¹ = loss.Σ⁻¹

    copyto!(Σ⁻¹, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "∑⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
    isnothing(objective) || (objective = ld + tr(Σ⁻¹Σₒ))

    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(loss))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            mul!(gradient, ∇Σ', vec(Σ⁻¹ - Σ⁻¹Σₒ * Σ⁻¹ - μ₋ᵀΣ⁻¹'μ₋ᵀΣ⁻¹))
            mul!(gradient, ∇μ', μ₋ᵀΣ⁻¹', -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J')
        end
        if !isnothing(hessian)
            if HessianEval(loss) === ApproxHessian
                mul!(hessian, ∇Σ' * kron(Σ⁻¹, Σ⁻¹), ∇Σ, 2, 0)
            else
                ∇²Σ = implied.∇²Σ
                # inner
                implied.∇²Σ_eval!(∇²Σ, J, par)
                # outer
                H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
                mul!(hessian, ∇Σ' * H_outer, ∇Σ)
                hessian .+= ∇²Σ
            end
        end
    end
    return objective
end

############################################################################################
### Non-Symbolic Implied Types

function evaluate!(objective, gradient, hessian, loss::SemML, par)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic implied type is not available")
    end

    implied = SEM.implied(loss)

    Σ = implied.Σ
    Σₒ = obs_cov(observed(loss))
    Σ⁻¹Σₒ = loss.Σ⁻¹Σₒ
    Σ⁻¹ = loss.Σ⁻¹
    copyto!(Σ⁻¹, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "Σ⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

    if !isnothing(objective)
        objective = ld + tr(Σ⁻¹Σₒ)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(loss))
            μ₋ = μₒ - μ
            objective += dot(μ₋, Σ⁻¹, μ₋)
        end
    end

    if !isnothing(gradient)
        S = implied.S
        F⨉I_A⁻¹ = implied.F⨉I_A⁻¹
        I_A⁻¹ = implied.I_A⁻¹
        ∇A = implied.∇A
        ∇S = implied.∇S

        C = F⨉I_A⁻¹' * (I - Σ⁻¹Σₒ) * Σ⁻¹ * F⨉I_A⁻¹
        mul!(gradient, ∇A', vec(C * S * I_A⁻¹'), 2, 0)
        mul!(gradient, ∇S', vec(C), 1, 1)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(loss))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
            mul!(gradient, ∇M', k', -2, 1)
            mul!(gradient, ∇A', vec(k' * (I_A⁻¹ * (M + S * k'))'), -2, 1)
            mul!(gradient, ∇S', vec(k'k), -1, 1)
        end
    end

    return objective
end

############################################################################################
### additional functions
############################################################################################

function non_posdef_return(par)
    if eltype(par) <: AbstractFloat
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
end

############################################################################################
### recommended methods
############################################################################################

update_observed(loss::SemML, observed::SemObservedMissing; kwargs...) =
    error("ML estimation does not work with missing data - use FIML instead")

function update_observed(loss::SemML, observed::SemObserved; kwargs...)
    if (obs_cov(loss) == obs_cov(observed)) && (obs_mean(loss) == obs_mean(observed))
        return loss # no change
    else
        return SemML(
            observed,
            loss.implied;
            approximate_hessian = HessianEval(loss) == ApproxHessian,
            kwargs...,
        )
    end
end
