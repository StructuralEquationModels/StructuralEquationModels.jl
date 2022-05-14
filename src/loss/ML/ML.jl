# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,M,M2,B, V} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    mult::M
    meandiff::M2
    approx_H::B
    has_meanstructure::V
end

############################################################################
### Constructors
############################################################################

function SemML(;observed, imply, n_par, approx_H = false, parameter_type = Float64, kwargs...)
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

function SemML_factorization!(Σ⁻¹, Σ)
    copyto!(Σ⁻¹, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    return Σ_chol
end

function SemML_ld_inv_mul!(Σ⁻¹, Σ⁻¹Σₒ, Σ_chol, Σₒ)
    ld = logdet(Σ_chol)
    Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
    mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
    return ld
end

function SemML_meandiff(μₒ, μ, Σ⁻¹)
    μ₋ = μₒ - μ
    μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
    return μ₋, μ₋ᵀΣ⁻¹
end

# first, dispatch for meanstructure
objective!(semml::SemML, par, model) = objective!(semml::SemML, par, model, semml.has_meanstructure)
gradient!(gradient, semml::SemML, par, model) = gradient!(gradient, semml::SemML, par, model, semml.has_meanstructure)
hessian!(hessian, semml::SemML, par, model) = hessian!(hessian, semml::SemML, par, model, semml.has_meanstructure)
objective_gradient!(gradient, semml::SemML, par, model) = objective_gradient!(gradient, semml::SemML, par, model, semml.has_meanstructure)
objective_hessian!(hessian, semml::SemML, par, model) = objective_hessian!(hessian, semml::SemML, par, model, semml.has_meanstructure)
gradient_hessian!(gradient, hessian, semml::SemML, par, model) = gradient_hessian!(gradient, hessian, semml::SemML, par, model, semml.has_meanstructure)
objective_gradient_hessian!(gradient, hessian, semml::SemML, par, model) = objective_gradient_hessian!(gradient, hessian, semml::SemML, par, model, semml.has_meanstructure)



function non_posdef_return(par)
    if eltype(par) <: AbstractFloat 
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
end

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  semml.mult, Σ⁻¹ = Σ⁻¹(semml)

        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        return ld + tr(Σ⁻¹Σₒ)
    end
end

#= function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  semml.mult, Σ⁻¹ = Σ⁻¹(semml)

        Σ_chol = SemML_factorization!(Σ⁻¹, Σ)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = SemML_ld_inv_mul!(Σ⁻¹, Σ⁻¹Σₒ, Σ_chol, Σₒ)

        return ld + tr(Σ⁻¹S)
    end
end =#



# μ₋ = μ_diff
# μ₋Σ⁻¹ = diff⨉inv

function objective!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  semml.mult, Σ⁻¹ = Σ⁻¹(semml),
        μ = μ(imply(model)), μₒ = obs_mean(observed(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)
        
        μ₋ = μₒ - μ

        return ld + tr(semml.mult) + dot(μ₋, Σ⁻¹, μ₋)
    end
end

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  semml.mult, Σ⁻¹ = Σ⁻¹(semml), ∇Σ = ∇Σ(imply(model))
        
        copyto!(Σ⁻¹, Σ)
        Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        Σ⁻¹ .= LinearAlgebra.inv!(Σ_chol)
        mul!(Σ⁻¹Σₒ, Σ⁻¹, Σₒ)

        return (vec(Σ⁻¹)-vec(Σ⁻¹Σₒ*Σ⁻¹))'*∇Σ
    end
end

function gradient!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}

    let Σ = Σ(imply(model)), ∇Σ = ∇Σ(imply(model)), Σₒ = obs_cov(observed(model)), Σ⁻¹Σₒ =  semml.mult, Σ⁻¹ = Σ⁻¹(semml),
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


function hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{false}) where {O, I <: SemImplySymbolic, L, D}

    # cholesky factorization
    copyto!(semml.inverses, Σ(imply(model)))
    a = cholesky!(Symmetric(semml.inverses); check = false)
    
    # ld, inverse, mult
    ld = logdet(a)
    semml.inverses .= LinearAlgebra.inv!(a)
    mul!(semml.mult, semml.inverses, obs_cov(observed(model)))

    if semml.approx_H
        hessian = 2*∇Σ(imply(model))'*kron(semml.inverses, semml.inverses)*∇Σ(imply(model))
    else
        J = (vec(semml.inverses)-vec(semml.mult*semml.inverses))'
        H_outer = 2*kron(semml.mult*semml.inverses, semml.inverses) - kron(semml.inverses, semml.inverses)
        hessian = ∇Σ(imply(model))'*H_outer*∇Σ(imply(model))
        ∇²Σ_function(imply(model))(∇²Σ(imply(model)), J, par)
        hessian += ∇²Σ(imply(model))
    end
    
    return hessian
end

function hessian!(semml::SemML, par, model::Sem{O, I, L, D}, has_meanstructure::Val{true}) where {O, I <: SemImplySymbolic, L, D}
    throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet"))
end


        # without means
        if isnothing(μ(imply(model)))

            if G && H
                J = (vec(semml.inverses)-vec(semml.inverses*obs_cov(observed(model))*semml.inverses))'
                gradient = J*∇Σ(imply(model))
                semml.gradient .= gradient'
                if semml.approx_H
                    hessian = 2*∇Σ(imply(model))'*kron(semml.inverses, semml.inverses)*∇Σ(imply(model))
                end
                if !semml.approx_H
                    M = semml.inverses*obs_cov(observed(model))*semml.inverses
                    H_outer = 
                        2*kron(M, semml.inverses) - 
                        kron(semml.inverses, semml.inverses)
                    hessian = ∇Σ(imply(model))'*H_outer*∇Σ(imply(model))
                    ∇²Σ_function(imply(model))(∇²Σ(imply(model)), J, par)
                    hessian += ∇²Σ(imply(model))
                end
                semml.hessian .= hessian
            end

            if G && !H
                gradient = (vec(semml.inverses)-vec(semml.inverses*obs_cov(observed(model))*semml.inverses))'*∇Σ(imply(model))
                semml.gradient .= gradient'
            end

            if !G && H
                J = (vec(semml.inverses)-vec(semml.inverses*obs_cov(observed(model))*semml.inverses))'
                if semml.approx_H
                    hessian = 2*∇Σ(imply(model))'*kron(semml.inverses, semml.inverses)*∇Σ(imply(model))
                end
                if !semml.approx_H
                    M = semml.inverses*obs_cov(observed(model))*semml.inverses
                    H_outer = 
                        2*kron(M, semml.inverses) - 
                        kron(semml.inverses, semml.inverses)
                    hessian = ∇Σ(imply(model))'*H_outer*∇Σ(imply(model))
                    ∇²Σ_function(imply(model))(∇²Σ(imply(model)), J, par)
                    hessian += ∇²Σ(imply(model)) 
                end
                semml.hessian .= hessian
            end

            if F
                mul!(semml.mult, semml.inverses, obs_cov(observed(model)))
                semml.objective[1] = ld + tr(semml.mult)
            end
        else
        # with means

            if H throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet")) end
            if G

            end
            if F

            end
        end
    end
end

# for non-symbolic imply type
function (semml::SemML)(par, F, G, H, model::Sem{O, I, L, D}) where {O, I <: RAM, L, D}

    if H
        throw(DomainError(H, "hessian for ML estimation with non-symbolic imply type is not implemented"))
    end

    semml.inverses .= Σ(imply(model))
    a = cholesky!(Symmetric(semml.inverses); check = false)

    if !isposdef(a)
        if G semml.gradient .= 1.0 end
        if H semml.hessian .= 1.0 end
        if F semml.objective[1] = Inf end
    else
        ld = logdet(a)
        semml.inverses .= LinearAlgebra.inv!(a)

        # without means
        if isnothing(μ(imply(model)))

            if G
                gradient = SemML_gradient(
                    S(imply(model)), 
                    F⨉I_A⁻¹(imply(model)), 
                    semml.inverses, 
                    I_A(imply(model)), 
                    ∇A(imply(model)), 
                    ∇S(imply(model)),
                    obs_cov(observed(model)))
                semml.gradient .= gradient'
            end

            if F
                mul!(semml.mult, semml.inverses, obs_cov(observed(model)))
                semml.objective[1] = ld + tr(semml.mult)
            end
            
        else # with means
            
            μ_diff = obs_mean(observed(model)) - μ(imply(model))
            diff⨉inv = μ_diff'*semml.inverses
            
            if G
                gradient = SemML_gradient(
                        S(imply(model)), 
                        F⨉I_A⁻¹(imply(model)), 
                        semml.inverses, 
                        I_A(imply(model)), 
                        ∇A(imply(model)), 
                        ∇S(imply(model)),
                        obs_cov(observed(model))) +
                    SemML_gradient_meanstructure(
                        diff⨉inv, 
                        F⨉I_A⁻¹(imply(model)), 
                        I_A(imply(model)), 
                        S(imply(model)),
                        M(imply(model)), 
                        ∇M(imply(model)), 
                        ∇A(imply(model)),
                        ∇S(imply(model)))
                semml.gradient .= gradient'
            end

            if F
                mul!(semml.mult, semml.inverses, obs_cov(observed(model)))
                semml.objective[1] = ld + tr(semml.mult) + diff⨉inv*μ_diff
            end

        end
    end
end

############################################################################
### recommended methods
############################################################################

objective(lossfun::SemML) = lossfun.objective
gradient(lossfun::SemML) = lossfun.gradient
hessian(lossfun::SemML) = lossfun.hessian

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
### additional functions
############################################################################

function SemML_gradient_common(F⨉I_A⁻¹, obs_cov, Σ⁻¹)
    M = transpose(F⨉I_A⁻¹)*transpose(LinearAlgebra.I-obs_cov*Σ⁻¹)*Σ⁻¹*F⨉I_A⁻¹
    return M
end

function SemML_gradient(S, F⨉I_A⁻¹, Σ⁻¹, I_A⁻¹, ∇A, ∇S, obs_cov)
    M = SemML_gradient_common(F⨉I_A⁻¹, obs_cov, Σ⁻¹)
    G = 2vec(M*S*I_A⁻¹')'∇A + vec(M)'∇S
    return G
end

function SemML_gradient_meanstructure(diff⨉inv, F⨉I_A⁻¹, I_A⁻¹, S, M, ∇M, ∇A, ∇S)
    k = diff⨉inv*F⨉I_A⁻¹
    G = -2k*∇M - 2vec(k'M'I_A⁻¹')'∇A - 2vec(k'k*S*I_A⁻¹')'∇A - vec(k'k)'∇S
    return G
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemML)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end