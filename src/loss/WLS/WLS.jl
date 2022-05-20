##### weighted least squares

############################################################################
### Types
############################################################################

struct SemWLS{Vt, St, B, C, B2} <: SemLossFunction
    V::Vt
    σₒ::St
    approx_H::B
    V_μ::C
    has_meanstructure::B2
end

############################################################################
### Constructors
############################################################################

function SemWLS(;observed, wls_weight_matrix = nothing, V_μ = nothing, approx_H = false, meanstructure = false, kwargs...)
    ind = CartesianIndices(obs_cov(observed))
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = obs_cov(observed)[ind]

    # compute V here
    if isnothing(wls_weight_matrix)
        D = duplication_matrix(n_man(observed))
        S = inv(obs_cov(observed))
        S = kron(S, S)
        V = 0.5*(D'*S*D)
    end

    if meanstructure
        if isnothing(V_μ)
            V_μ = inv(obs_cov(observed))
        end
    else
        V_μ = nothing
    end

    return SemWLS(
        V, 
        s, 
        approx_H, 
        V_μ,
        Val(meanstructure)
    )
end

############################################################################
### methods
############################################################################

objective!(semwls::SemWLS, par, model::AbstractSemSingle) = objective!(semwls::SemWLS, par, model, semwls.has_meanstructure)
gradient!(semwls::SemWLS, par, model::AbstractSemSingle) = gradient!(semwls::SemWLS, par, model, semwls.has_meanstructure)
hessian!(semwls::SemWLS, par, model::AbstractSemSingle) = hessian!(semwls::SemWLS, par, model, semwls.has_meanstructure)

objective_gradient!(semwls::SemWLS, par, model::AbstractSemSingle) = objective_gradient!(semwls::SemWLS, par, model, semwls.has_meanstructure)
objective_hessian!(semwls::SemWLS, par, model::AbstractSemSingle) = objective_hessian!(semwls::SemWLS, par, model, semwls.has_meanstructure)
gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle) = gradient_hessian!(semwls::SemWLS, par, model, semwls.has_meanstructure)

objective_gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle) = objective_gradient_hessian!(semwls::SemWLS, par, model, semwls.has_meanstructure)


function objective!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{T}) where T
    
    let σ = Σ(imply(model)), μ = μ(imply(model)), σₒ = semwls.σₒ, μₒ = obs_mean(observed(model)), V = semwls.V, V_μ = semwls.V_μ, 
        
        σ₋ = σₒ - σ
        
        if T
            μ₋ = μₒ - μ
            return dot(σ₋, V, σ₋) + dot(μ₋, V_μ, μ₋)
        else
            return dot(σ₋, V, σ₋)  
        end
    end
end

function gradient!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{T}) where T
    
    let σ = Σ(imply(model)), μ = μ(imply(model)), σₒ = semwls.σₒ, μₒ = obs_mean(observed(model)), V = semwls.V, V_μ = semwls.V_μ,
        ∇σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model))
        
        σ₋ = σₒ - σ
        
        if T
            μ₋ = μₒ - μ
            return -2*(σ₋'*V*∇σ + μ₋'*V_μ*∇μ)'
        else
            return -2*(σ₋'*V*∇σ)'
        end
    end
end

function hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{T}) where T
    
    let σ = Σ(imply(model)), σₒ = semwls.σₒ, V = semwls.V,
        ∇σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        σ₋ = σₒ - σ
        
        if T
            throw(DomainError(H, "hessian of WLS with meanstructure is not available"))
        else
            hessian = 2*∇σ'*V*∇σ
            if !semwls.approx_H
                J = -2*(σ₋'*semwls.V)'
                ∇²Σ_function!(∇²Σ, J, par)
                hessian += ∇²Σ
            end
            return hessian
        end
    end
end

function objective_gradient!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{T}) where T
    
    let σ = Σ(imply(model)), μ = μ(imply(model)), σₒ = semwls.σₒ, μₒ = obs_mean(observed(model)), V = semwls.V, V_μ = semwls.V_μ,
        ∇σ = ∇Σ(imply(model)), ∇μ = ∇μ(imply(model))
        
        σ₋ = σₒ - σ
        
        if T
            μ₋ = μₒ - μ
            objective = dot(σ₋, V, σ₋) + dot(μ₋', V_μ, μ₋)
            gradient = -2*(σ₋'*V*∇σ + μ₋'*V_μ*∇μ)'
            return objective, gradient
        else
            objective = dot(σ₋, V, σ₋) 
            gradient = -2*(σ₋'*V*∇σ)'
            return objective, gradient
        end
    end
end

function objective_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{T}) where T
    
    let σ = Σ(imply(model)), σₒ = semwls.σₒ, V = semwls.V, 
        ∇σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        σ₋ = σₒ - σ
        
        objective = dot(σ₋, V, σ₋)

        hessian = 2*∇σ'*V*∇σ
        if !semwls.approx_H
            J = -2*(σ₋'*semwls.V)'
            ∇²Σ_function!(∇²Σ, J, par)
            hessian += ∇²Σ
        end

        return objective, hessian
    end
end

objective_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{true}) =
    throw(DomainError(H, "hessian of WLS with meanstructure is not available"))

function gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{false})
    
    let σ = Σ(imply(model)), σₒ = semwls.σₒ, V = semwls.V,
        ∇σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        σ₋ = σₒ - σ
        
        gradient = -2*(σ₋'*V*∇σ)'

        hessian = 2*∇σ'*V*∇σ
        if !semwls.approx_H
            J = -2*(σ₋'*semwls.V)'
            ∇²Σ_function!(∇²Σ, J, par)
            hessian += ∇²Σ
        end

        return gradient, hessian
    end
end

gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{true}) =
    throw(DomainError(H, "hessian of WLS with meanstructure is not available"))

function objective_gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{false})
    
    let σ = Σ(imply(model)), σₒ = semwls.σₒ, V = semwls.V,
        ∇σ = ∇Σ(imply(model)),
        ∇²Σ_function! = ∇²Σ_function(imply(model)), ∇²Σ = ∇²Σ(imply(model))
        
        σ₋ = σₒ - σ
        
        objective = dot(σ₋, V, σ₋) 
        gradient = -2*(σ₋'*V*∇σ)'
        hessian = 2*∇σ'*V*∇σ
        if !semwls.approx_H
            J = -2*(σ₋'*semwls.V)'
            ∇²Σ_function!(∇²Σ, J, par)
            hessian += ∇²Σ
        end
        return objective, gradient, hessian
    end
end

objective_gradient_hessian!(semwls::SemWLS, par, model::AbstractSemSingle, has_meanstructure::Val{true}) =
    throw(DomainError(H, "hessian of WLS with meanstructure is not available"))

############################################################################
### Recommended methods
############################################################################

update_observed(lossfun::SemWLS, observed::SemObs; kwargs...) = SemWLS(;observed = observed, kwargs...)