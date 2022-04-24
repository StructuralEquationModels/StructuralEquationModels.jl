# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,M,M2,B,FT,GT,HT} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    mult::M
    meandiff::M2
    approx_H::B

    objective::FT
    gradient::GT
    hessian::HT
end

############################################################################
### Constructors
############################################################################

function SemML(;observed, n_par, approx_H = false, parameter_type = Float64, kwargs...)
    isnothing(obs_mean(observed)) ?
        meandiff = nothing :
        meandiff = copy(obs_mean(observed))
    return SemML(
        copy(obs_cov(observed)),
        copy(obs_cov(observed)),
        meandiff,
        approx_H,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par)
        )
end

############################################################################
### functors
############################################################################

# for symbolic imply type
function (semml::SemML)(
    par, 
    F, 
    G, 
    H, 
    model::Sem{O, I, L, D}) where {O, I <: SemImplySymbolic, L, D}
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
        μ_diff = obs_mean(observed(model)) - μ(imply(model))
        diff⨉inv = μ_diff'*semml.inverses
            if H throw(DomainError(H, "hessian of ML + meanstructure is not implemented yet")) end
            if G
                gradient = 
                    vec(
                        semml.inverses*(
                            LinearAlgebra.I - 
                            obs_cov(observed(model))*semml.inverses - 
                            μ_diff*diff⨉inv))'*∇Σ(imply(model)) -
                    2*diff⨉inv*∇μ(imply(model))
                semml.gradient .= gradient'
            end
            if F
                mul!(semml.mult, semml.inverses, obs_cov(observed(model)))
                semml.objective[1] = ld + tr(semml.mult) + diff⨉inv*μ_diff
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