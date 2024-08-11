# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(;observed, meanstructure = false, approximate_hessian = false, kwargs...)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `meanstructure::Bool`: does the model have a meanstructure?
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(observed = my_observed)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure, also analytic hessians.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemML{HE <: HessianEvaluation, INV, M, M2} <: SemLossFunction{HE}
    Σ⁻¹::INV
    Σ⁻¹Σₒ::M
    meandiff::M2

    SemML{HE}(args...) where {HE <: HessianEvaluation} =
        new{HE, map(typeof, args)...}(args...)
end

############################################################################################
### Constructors
############################################################################################

function SemML(; observed::SemObserved, approximate_hessian::Bool = false, kwargs...)
    obsmean = obs_mean(observed)
    obscov = obs_cov(observed)
    meandiff = isnothing(obsmean) ? nothing : copy(obsmean)

    return SemML{approximate_hessian ? ApproximateHessian : ExactHessian}(
        similar(obscov),
        similar(obscov),
        meandiff,
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Imply Types

function evaluate!(
    objective,
    gradient,
    hessian,
    semml::SemML,
    implied::SemImplySymbolic,
    model::AbstractSemSingle,
    par,
)
    if !isnothing(hessian)
        (MeanStructure(implied) === HasMeanStructure) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(model))
    Σ⁻¹Σₒ = semml.Σ⁻¹Σₒ
    Σ⁻¹ = semml.Σ⁻¹

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

    if MeanStructure(implied) === HasMeanStructure
        μ = implied.μ
        μₒ = obs_mean(observed(model))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            gradient .= (vec(Σ⁻¹ - Σ⁻¹Σₒ * Σ⁻¹ - μ₋ᵀΣ⁻¹'μ₋ᵀΣ⁻¹)' * ∇Σ)'
            gradient .-= (2 * μ₋ᵀΣ⁻¹ * ∇μ)'
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            gradient .= (J * ∇Σ)'
        end
        if !isnothing(hessian)
            if HessianEvaluation(semml) === ApproximateHessian
                mul!(hessian, 2 * ∇Σ' * kron(Σ⁻¹, Σ⁻¹), ∇Σ)
            else
                ∇²Σ_function! = implied.∇²Σ_function
                ∇²Σ = implied.∇²Σ
                # inner
                ∇²Σ_function!(∇²Σ, J, par)
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
### Non-Symbolic Imply Types

function evaluate!(
    objective,
    gradient,
    hessian,
    semml::SemML,
    implied::RAM,
    model::AbstractSemSingle,
    par,
)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic imply type is not available")
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(model))
    Σ⁻¹Σₒ = semml.Σ⁻¹Σₒ
    Σ⁻¹ = semml.Σ⁻¹

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

        if MeanStructure(implied) === HasMeanStructure
            μ = implied.μ
            μₒ = obs_mean(observed(model))
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
        gradᵀ = 2vec(C * S * I_A⁻¹')'∇A + vec(C)'∇S

        if MeanStructure(implied) === HasMeanStructure
            μ = implied.μ
            μₒ = obs_mean(observed(model))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
            gradᵀ .+= -2k * ∇M - 2vec(k' * (M' + k * S) * I_A⁻¹')'∇A - vec(k'k)'∇S
        end
        copyto!(gradient, gradᵀ')
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

update_observed(lossfun::SemML, observed::SemObservedMissing; kwargs...) =
    error("ML estimation does not work with missing data - use FIML instead")

function update_observed(lossfun::SemML, observed::SemObserved; kwargs...)
    if size(lossfun.Σ⁻¹) == size(obs_cov(observed))
        return lossfun
    else
        return SemML(; observed = observed, kwargs...)
    end
end
