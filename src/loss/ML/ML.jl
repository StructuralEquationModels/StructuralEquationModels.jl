# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,M,M2,B, V} <: SemLossFunction
    Σ⁻¹::INV 
    Σ⁻¹Σₒ::M
    meandiff::M2
    approx_H::B
    has_meanstructure::V
end

############################################################################
### Constructors
############################################################################

function SemML(;observed, imply, approx_H = false, kwargs...)
    isnothing(obs_mean(observed)) ?
        meandiff = nothing :
        meandiff = copy(obs_mean(observed))
    return SemML(
        copy(obs_cov(observed)),
        copy(obs_cov(observed)),
        meandiff,
        Val(approx_H),
        has_meanstructure(imply)
        )
end

############################################################################
### objective, gradient, hessian methods
############################################################################

# first, dispatch for meanstructure
objective!(semml::SemML, par, model) = objective!(semml::SemML, par, model, semml.has_meanstructure)
gradient!(semml::SemML, par, model) = gradient!(semml::SemML, par, model, semml.has_meanstructure)
hessian!(semml::SemML, par, model) = hessian!(semml::SemML, par, model, semml.has_meanstructure)
objective_gradient!(semml::SemML, par, model) = objective_gradient!(semml::SemML, par, model, semml.has_meanstructure)
objective_hessian!(semml::SemML, par, model) = objective_hessian!(semml::SemML, par, model, semml.has_meanstructure)
gradient_hessian!(semml::SemML, par, model) = gradient_hessian!(semml::SemML, par, model, semml.has_meanstructure)
objective_gradient_hessian!(semml::SemML, par, model) = objective_gradient_hessian!(semml::SemML, par, model, semml.has_meanstructure)

############################################################################
### Symbolic Imply Types

# objective -----------------------------------------------------------------------------------------------------------------------------

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml)

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        return ld + tr(Σ⁻¹Σₒ)
    end
end

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        
        μ₋ = μₒ - μ

        return ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
    end
end

# gradient -----------------------------------------------------------------------------------------------------------------------------

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), ∇Σ = ∇Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        gradient = (vec(Σ⁻¹)-vec(Σ⁻¹Σₒ*Σ⁻¹))'*∇Σ
        return gradient'
    end
end

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), ∇Σ = ∇Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), ∇μ = ∇μ(imply(model)), μₒ = obs_mean(observed(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        μ₋ = μₒ - μ
        μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹

        gradient = (vec(Σ⁻¹*(I - Σ⁻¹Σₒ - μ₋*μ₋ᵀΣ⁻¹))'*∇Σ - 2*μ₋ᵀΣ⁻¹*∇μ)'
    
        return gradient
    end
end

# hessian -----------------------------------------------------------------------------------------------------------------------------

function hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), ∇Σ = ∇Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))

    copyto!(Σ⁻¹, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

    Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)

    if semml.approx_H
        hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
    else
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
        # inner
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        ∇²Σ_function!(∇²Σ, J, par)
        # outer
        H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
        hessian = ∇Σ'*H_outer*∇Σ
        hessian += ∇²Σ
    end
    
    return hessian
    end
end

# no hessian for models with meanstructures
function hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

# objective_gradient -----------------------------------------------------------------------------------------------------------------------------

function objective_gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        gradient = (vec(Σ⁻¹)-vec(Σ⁻¹Σₒ*Σ⁻¹))'*∇Σ

        return objective, gradient'
    end
end

function objective_gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model)), 
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            μ₋ = μₒ - μ
            objective = ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
        end
        
        μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
        gradient = (vec(Σ⁻¹*(I - Σ⁻¹Σₒ - μ₋*μ₋ᵀΣ⁻¹))'*∇Σ - 2*μ₋ᵀΣ⁻¹*∇μ)'

        return objective, gradient
    end
end

# objective_hessian ------------------------------------------------------------------------------------------------------------------------------

function objective_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
            # inner
            J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end

        return objective, hessian
    end
end

function objective_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

# gradient_hessian -------------------------------------------------------------------------------------------------------------------------------

function gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), 
        ∇Σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J*∇Σ

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end
        
        return gradient, hessian
    end
end

function gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

# objective_gradient_hessian ---------------------------------------------------------------------------------------------------------------------

function objective_gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml), ∇Σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        gradient = J*∇Σ

        if semml.approx_H
            hessian = 2*∇Σ'*kron(Σ⁻¹, Σ⁻¹)*∇Σ
        else
            Σ⁻¹ΣₒΣ⁻¹ = Σ⁻¹Σₒ*Σ⁻¹
            # inner
            ∇²Σ_function!(∇²Σ, J, par)
            # outer
            H_outer = 2*kron(Σ⁻¹ΣₒΣ⁻¹, Σ⁻¹) - kron(Σ⁻¹, Σ⁻¹)
            hessian = ∇Σ'*H_outer*∇Σ
            hessian += ∇²Σ
        end
        
        return objective, gradient, hessian
    end
end

function objective_gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

############################################################################
### Non-Symbolic Imply Types

# no hessians ---------------------------------------------------------------------------------------------------------------------

function hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure) where {O, I <: RAM, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

function objective_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure) where {O, I <: RAM, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

function gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure) where {O, I <: RAM, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

function objective_gradient_hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure) where {O, I <: RAM, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end

# objective ----------------------------------------------------------------------------------------------------------------------

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: RAM, L, D}
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml)

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        return ld + tr(Σ⁻¹Σₒ)
    end
end

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: RAM, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        
        μ₋ = μₒ - μ

        return ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
    end
end

# gradient -----------------------------------------------------------------------------------------------------------------------

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: RAM, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model)), I = LinearAlgebra.I
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        M = F⨉I_A⁻¹'*(I-Σ⁻¹Σₒ)'*Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S
        
        return gradient'
    end
end

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: RAM, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model)), ∇M = ∇M(imply(model)),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        M = F⨉I_A⁻¹'*(I-Σ⁻¹Σₒ)'Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S

        μ₋ = μₒ - μ
        μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹

        k = μ₋ᵀΣ⁻¹*F⨉I_A⁻¹
        gradient_mean = -2k*∇M - 2vec(k'M'I_A⁻¹')'∇A - 2vec(k'k*S*I_A⁻¹')'∇A - vec(k'k)'∇S

        return (gradient + gradient_mean)'
    end
end

# objective_gradient -------------------------------------------------------------------------------------------------------------

function objective_gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: RAM, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            objective = ld + tr(Σ⁻¹Σₒ)
        end

        M = F⨉I_A⁻¹'*(I-Σ⁻¹Σₒ)'*Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S
        
        return objective, gradient'
    end
end

function objective_gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: RAM, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  Σ⁻¹Σₒ(semml), Σ⁻¹ = Σ⁻¹(semml),
        S = S(imply(model)), F⨉I_A⁻¹ = F⨉I_A⁻¹(imply(model)), I_A⁻¹ = I_A⁻¹(imply(model)), 
        ∇A = ∇A(imply(model)), ∇S = ∇S(imply(model)), ∇M = ∇M(imply(model)),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) 
            objective = non_posdef_return(par) 
        else
            ld = logdet(Σ_chol)
            Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
            mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
            μ₋ = μₒ - μ
            objective = ld + tr(Σ⁻¹Σₒ(semml)) + dot(μ₋, Σ⁻¹, μ₋)
        end

        M = F⨉I_A⁻¹'*(I-Σ⁻¹Σₒ)'Σ⁻¹*F⨉I_A⁻¹
        gradient = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S

        μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹

        k = μ₋ᵀΣ⁻¹*F⨉I_A⁻¹
        gradient_mean = -2k*∇M - 2vec(k'M'I_A⁻¹')'∇A - 2vec(k'k*S*I_A⁻¹')'∇A - vec(k'k)'∇S

        gradient = (gradient + gradient_mean)'

        return objective, gradient
    end
end

############################################################################
### recommended methods
############################################################################

update_observed(lossfun::SemML, observed::SemObsMissing; kwargs...) = 
    throw(ArgumentError("ML estimation does not work with missing data - use FIML instead"))

function update_observed(lossfun::SemML, observed::SemObs; kwargs...)
    if (size(lossfun.inverses) == size(obs_cov(observed))) & (isnothing(lossfun.meandiff) == isnothing(obs_mean(observed)))
        return lossfun
    else
        return SemML(;observed = observed, kwargs...)
    end
end

############################################################################
### additional methods
############################################################################

Σ⁻¹(semml::SemML) = semml.Σ⁻¹
Σ⁻¹Σₒ(semml::SemML) = semml.Σ⁻¹Σₒ

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemML)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end