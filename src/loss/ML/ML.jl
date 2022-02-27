# Ordinary Maximum Likelihood Estimation

############################################################################
### Types
############################################################################

struct SemML{INV,C,L,M,M2,B,FT,GT,HT} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    mult::M
    logdets::L #logdets of implied covmats
    meandiff::M2
    approx_H::B

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemML(;observed, n_par, approx_H = false, parameter_type = Float64, kwargs...)
    isnothing(observed.obs_mean) ?
        meandiff = nothing :
        meandiff = copy(observed.obs_mean)
    return SemML(
        copy(observed.obs_cov),
        nothing,
        copy(observed.obs_cov),
        nothing,
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
    semml.inverses .= model.imply.Σ
    a = cholesky!(Symmetric(semml.inverses); check = false)

    if !isposdef(a)
        if G semml.G .= 0.0 end
        if H semml.H .= 0.0 end
        if F semml.F[1] = Inf end
    else
        ld = logdet(a)
        semml.inverses .= LinearAlgebra.inv!(a)

        # without means
        if isnothing(model.imply.μ)

            if G && H
                J = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'
                gradient = J*model.imply.∇Σ
                semml.G .= gradient'
                if semml.approx_H
                    hessian = 2*model.imply.∇Σ'*kron(semml.inverses, semml.inverses)*model.imply.∇Σ
                end
                if !semml.approx_H
                    M = semml.inverses*model.observed.obs_cov*semml.inverses
                    H_outer = 
                        2*kron(M, semml.inverses) - 
                        kron(semml.inverses, semml.inverses)
                    hessian = model.imply.∇Σ'*H_outer*model.imply.∇Σ
                    model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                    hessian += model.imply.∇²Σ
                end
                semml.H .= hessian
            end

            if G && !H
                gradient = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'*model.imply.∇Σ
                semml.G .= gradient'
            end

            if !G && H
                J = (vec(semml.inverses)-vec(semml.inverses*model.observed.obs_cov*semml.inverses))'
                if semml.approx_H
                    hessian = 2*model.imply.∇Σ'*kron(semml.inverses, semml.inverses)*model.imply.∇Σ
                end
                if !semml.approx_H
                    M = semml.inverses*model.observed.obs_cov*semml.inverses
                    H_outer = 
                        2*kron(M, semml.inverses) - 
                        kron(semml.inverses, semml.inverses)
                    hessian = model.imply.∇Σ'*H_outer*model.imply.∇Σ
                    model.imply.∇²Σ_function(model.imply.∇²Σ, J, par)
                    hessian += model.imply.∇²Σ 
                end
                semml.H .= hessian
            end

            if F
                mul!(semml.mult, semml.inverses, model.observed.obs_cov)
                semml.F[1] = ld + tr(semml.mult)
            end
        else
        # with means
        μ_diff = model.observed.obs_mean - model.imply.μ
        diff⨉inv = μ_diff'*semml.inverses
            if H stop("hessian of ML + meanstructure is not implemented yet") end
            if G
                gradient = 
                    vec(
                        semml.inverses*(
                            LinearAlgebra.I - 
                            model.observed.obs_cov*semml.inverses - 
                            μ_diff*diff⨉inv))'*model.imply.∇Σ -
                    2*diff⨉inv*model.imply.∇μ
                semml.G .= gradient'
            end
            if F
                mul!(semml.mult, semml.inverses, model.observed.obs_cov)
                semml.F[1] = ld + tr(semml.mult) + diff⨉inv*μ_diff
            end
        end
    end
end

# for non-symbolic imply type
function (semml::SemML)(par, F, G, H, model::Sem{O, I, L, D}) where {O, I <: RAM, L, D}

    if H
        stop("Hessian for ML estimation with non-symbolic imply type is not implemented")
    end

    semml.inverses .= model.imply.Σ
    a = cholesky!(Symmetric(semml.inverses); check = false)

    if !isposdef(a)
        if G semml.G .= 0.0 end
        if H semml.H .= 0.0 end
        if F semml.F[1] = Inf end
    else
        ld = logdet(a)
        semml.inverses .= LinearAlgebra.inv!(a)

        # without means
        if isnothing(model.imply.μ)

            if G
                gradient = SemML_gradient(
                    model.imply.S, 
                    model.imply.F⨉I_A⁻¹, 
                    semml.inverses, 
                    model.imply.I_A, 
                    model.imply.∇A, 
                    model.imply.∇S,
                    model.observed.obs_cov)
                semml.G .= gradient'
            end

            if F
                mul!(semml.mult, semml.inverses, model.observed.obs_cov)
                semml.F[1] = ld + tr(semml.mult)
            end
            
        else
        # with means
        μ_diff = model.observed.obs_mean - model.imply.μ
        diff⨉inv = μ_diff'*semml.inverses
            
            if G
                gradient = SemML_gradient(
                        model.imply.S, 
                        model.imply.F⨉I_A⁻¹, 
                        semml.inverses, 
                        model.imply.I_A, 
                        model.imply.∇A, 
                        model.imply.∇S,
                        model.observed.obs_cov) +
                    SemML_gradient_meanstructure(
                        diff⨉inv, 
                        model.imply.F⨉I_A⁻¹, 
                        model.imply.I_A, 
                        model.imply.S,
                        model.imply.M, 
                        model.imply.∇M, 
                        model.imply.∇A,
                        model.imply.∇S)
                semml.G .= gradient'
            end

            if F
                mul!(semml.mult, semml.inverses, model.observed.obs_cov)
                semml.F[1] = ld + tr(semml.mult) + diff⨉inv*μ_diff
            end

        end
    end
end

############################################################################
### additional functions
############################################################################

#= function SemML_gradient_A(F⨉I_A⁻¹, S, Σ⁻¹, Ω, ∇A)
    2*vec()
end =#

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