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
struct SemML{INV, M, M2, B, V} <: SemLossFunction
    Σ⁻¹::INV
    Σ⁻¹Σₒ::M
    meandiff::M2
    approximate_hessian::B
    has_meanstructure::V
end

############################################################################################
### Constructors
############################################################################################

function SemML(; observed, meanstructure = false, approximate_hessian = false, kwargs...)
    isnothing(obs_mean(observed)) ? meandiff = nothing : meandiff = copy(obs_mean(observed))
    return SemML(
        similar(obs_cov(observed)),
        similar(obs_cov(observed)),
        meandiff,
        approximate_hessian,
        Val(meanstructure),
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

# first, dispatch for meanstructure
objective!(semml::SemML, par, model::AbstractSemSingle) =
    objective!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
gradient!(semml::SemML, par, model::AbstractSemSingle) =
    gradient!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
hessian!(semml::SemML, par, model::AbstractSemSingle) =
    hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_gradient!(semml::SemML, par, model::AbstractSemSingle) =
    objective_gradient!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_hessian!(semml::SemML, par, model::AbstractSemSingle) =
    objective_hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
gradient_hessian!(semml::SemML, par, model::AbstractSemSingle) =
    gradient_hessian!(semml::SemML, par, model, semml.has_meanstructure, imply(model))
objective_gradient_hessian!(semml::SemML, par, model::AbstractSemSingle) =
    objective_gradient_hessian!(
        semml::SemML,
        par,
        model,
        semml.has_meanstructure,
        imply(model),
    )

############################################################################################
### Symbolic Imply Types

function objective!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::SemImplySymbolic,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)),
        μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) || return non_posdef_return(par)
        ld = logdet(Σ_chol)
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        #mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            return ld + dot(Σ⁻¹, Σₒ) + dot(μ₋, Σ⁻¹, μ₋)
        else
            return ld + dot(Σ⁻¹, Σₒ)
        end
    end
end

function gradient!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::SemImplySymbolic,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)),
        μ = μ(imply(model)),
        ∇μ = ∇μ(imply(model)),
        μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) || return ones(eltype(par), size(par))
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            gradient = vec(Σ⁻¹ - Σ⁻¹Σₒ * Σ⁻¹ - μ₋ᵀΣ⁻¹'μ₋ᵀΣ⁻¹)' * ∇Σ - 2 * μ₋ᵀΣ⁻¹ * ∇μ
            return gradient'
        else
            gradient = vec(Σ⁻¹ - Σ⁻¹Σₒ * Σ⁻¹)' * ∇Σ
            return gradient'
        end
    end
end

function hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{false},
    imp::SemImplySymbolic,
)
    let Σ = Σ(imply(model)),
        ∇Σ = ∇Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        ∇²Σ_function! = ∇²Σ_function(imply(model)),
        ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) || return diagm(fill(one(eltype(par)), length(par)))
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)

        if semml.approximate_hessian
            hessian = 2 * ∇Σ' * kron(Σ⁻¹, Σ⁻¹) * ∇Σ
        else
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹
            # inner
            J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ' * H_outer * ∇Σ
            hessian .+= ∇²Σ
        end

        return hessian
    end
end

function hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{true},
    imp::SemImplySymbolic,
)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

function objective_gradient!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::SemImplySymbolic,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)),
        μₒ = obs_mean(observed(model)),
        ∇Σ = ∇Σ(imply(model)),
        ∇μ = ∇μ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        if !isposdef(Σ_chol)
            return non_posdef_return(par), ones(eltype(par), size(par))
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

            if T
                μ₋ = μₒ - μ
                μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹

                objective = ld + tr(Σ⁻¹Σₒ) + dot(μ₋, Σ⁻¹, μ₋)
                gradient = vec(Σ⁻¹ * (I - Σₒ * Σ⁻¹ - μ₋ * μ₋ᵀΣ⁻¹))' * ∇Σ - 2 * μ₋ᵀΣ⁻¹ * ∇μ
                return objective, gradient'
            else
                objective = ld + tr(Σ⁻¹Σₒ)
                gradient = (vec(Σ⁻¹) - vec(Σ⁻¹Σₒ * Σ⁻¹))' * ∇Σ
                return objective, gradient'
            end
        end
    end
end

function objective_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::SemImplySymbolic,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)),
        ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)),
        ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        if !isposdef(Σ_chol)
            return non_posdef_return(par), diagm(fill(one(eltype(par)), length(par)))
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)

            if semml.approximate_hessian
                hessian = 2 * ∇Σ' * kron(Σ⁻¹, Σ⁻¹) * ∇Σ
            else
                Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹
                # inner
                J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
                ∇²Σ_function!(∇²Σ, J, par)
                # outer
                H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
                hessian = ∇Σ' * H_outer * ∇Σ
                hessian .+= ∇²Σ
            end

            return objective, hessian
        end
    end
end

function objective_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{true},
    imp::SemImplySymbolic,
)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

function gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{false},
    imp::SemImplySymbolic,
)
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)),
        ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)),
        ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) ||
            return ones(eltype(par), size(par)), diagm(fill(one(eltype(par)), length(par)))
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J * ∇Σ

        if semml.approximate_hessian
            hessian = 2 * ∇Σ' * kron(Σ⁻¹, Σ⁻¹) * ∇Σ
        else
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ' * H_outer * ∇Σ
            hessian .+= ∇²Σ
        end

        return gradient', hessian
    end
end

function gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{true},
    imp::SemImplySymbolic,
)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

function objective_gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{false},
    imp::SemImplySymbolic,
)
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)),
        ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        if !isposdef(Σ_chol)
            objective = non_posdef_return(par)
            gradient = ones(eltype(par), size(par))
            hessian = diagm(fill(one(eltype(par)), length(par)))
            return objective, gradient, hessian
        end
        ld = logdet(Σ_chol)
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        objective = ld + tr(Σ⁻¹Σₒ)

        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J * ∇Σ

        if semml.approximate_hessian
            hessian = 2 * ∇Σ' * kron(Σ⁻¹, Σ⁻¹) * ∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ * Σ⁻¹
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ' * H_outer * ∇Σ
            hessian .+= ∇²Σ
        end

        return objective, gradient', hessian
    end
end

function objective_gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{true},
    imp::SemImplySymbolic,
)
    throw(DomainError(H, "hessian of ML + meanstructure is not available"))
end

############################################################################################
### Non-Symbolic Imply Types

# no hessians ------------------------------------------------------------------------------

function hessian!(semml::SemML, par, model::AbstractSemSingle, has_meanstructure, imp::RAM)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function objective_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure,
    imp::RAM,
)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure,
    imp::RAM,
)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

function objective_gradient_hessian!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure,
    imp::RAM,
)
    throw(DomainError(H, "hessian of ML + non-symbolic imply type is not available"))
end

# objective, gradient ----------------------------------------------------------------------

function objective!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::RAM,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)),
        μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) || return non_posdef_return(par)
        ld = logdet(Σ_chol)
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        if T
            μ₋ = μₒ - μ
            return ld + tr(Σ⁻¹Σₒ) + dot(μ₋, Σ⁻¹, μ₋)
        else
            return ld + tr(Σ⁻¹Σₒ)
        end
    end
end

function gradient!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::RAM,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)),
        M = M(imply(model)),
        F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)),
        I_A⁻¹ = I_A⁻¹(imply(model)),
        ∇A = ∇A(imply(model)),
        ∇S = ∇S(imply(model)),
        ∇M = ∇M(imply(model)),
        μ = μ(imply(model)),
        μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        isposdef(Σ_chol) || return ones(eltype(par), size(par))
        Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        C = F⨉I_A⁻¹' * (I - Σ⁻¹Σₒ) * Σ⁻¹ * F⨉I_A⁻¹
        gradient = 2vec(C * S * I_A⁻¹')'∇A + vec(C)'∇S

        if T
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹

            gradient .+= -2k * ∇M - 2vec(k' * (M' + k * S) * I_A⁻¹')'∇A - vec(k'k)'∇S
        end

        return gradient'
    end
end

function objective_gradient!(
    semml::SemML,
    par,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
    imp::RAM,
) where {T}
    let Σ = Σ(imply(model)),
        Σₒ = obs_cov(observed(model)),
        Σ⁻¹Σₒ = Σ⁻¹Σₒ(semml),
        Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)),
        M = M(imply(model)),
        F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)),
        I_A⁻¹ = I_A⁻¹(imply(model)),
        ∇A = ∇A(imply(model)),
        ∇S = ∇S(imply(model)),
        ∇M = ∇M(imply(model)),
        μ = μ(imply(model)),
        μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
        if !isposdef(Σ_chol)
            objective = non_posdef_return(par)
            gradient = ones(eltype(par), size(par))
            return objective, gradient
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)

            C = F⨉I_A⁻¹' * (I - Σ⁻¹Σₒ) * Σ⁻¹ * F⨉I_A⁻¹
            gradient = 2vec(C * S * I_A⁻¹')'∇A + vec(C)'∇S

            if T
                μ₋ = μₒ - μ
                objective += dot(μ₋, Σ⁻¹, μ₋)

                μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
                k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
                gradient .+= -2k * ∇M - 2vec(k' * (M' + k * S) * I_A⁻¹')'∇A - vec(k'k)'∇S
            end

            return objective, gradient'
        end
    end
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
    throw(ArgumentError("ML estimation does not work with missing data - use FIML instead"))

function update_observed(lossfun::SemML, observed::SemObserved; kwargs...)
    if size(lossfun.Σ⁻¹) == size(obs_cov(observed))
        return lossfun
    else
        return SemML(; observed = observed, kwargs...)
    end
end

############################################################################################
### additional methods
############################################################################################

Σ⁻¹(semml::SemML) = semml.Σ⁻¹
Σ⁻¹Σₒ(semml::SemML) = semml.Σ⁻¹Σₒ
