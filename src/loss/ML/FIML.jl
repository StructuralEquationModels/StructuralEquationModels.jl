############################################################################
### Types
############################################################################

mutable struct SemFIML{INV, C, L, O, M, IM, I, T, U, W, FT, GT, HT} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats

    ∇ind::O

    imp_mean::IM
    meandiff::M
    imp_inv::I

    mult::T

    commutation_indices::U

    interaction::W

    objective::FT
    gradient::GT
    hessian::HT
end

############################################################################
### Constructors
############################################################################

function SemFIML(;observed, specification, n_par, parameter_type = Float64, kwargs...)

    inverses = broadcast(x -> zeros(x, x), Int64.(pattern_nvar_obs(observed)))
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))

    n_patterns = size(rows(observed), 1)
    logdets = zeros(n_patterns)

    imp_mean = zeros.(Int64.(pattern_nvar_obs(observed)))
    meandiff = zeros.(Int64.(pattern_nvar_obs(observed)))

    nman = Int64(n_man(observed))
    imp_inv = zeros(nman, nman)
    mult = similar.(inverses)
    
    ∇ind = vec(CartesianIndices(Array{Float64}(undef, nman, nman)))
    ∇ind = [findall(x -> !(x[1] ∈ ind || x[2] ∈ ind), ∇ind) for ind in patterns_not(observed)]

    commutation_indices = get_commutation_lookup(get_n_nodes(specification)^2)

    return SemFIML(
    inverses,
    choleskys,
    logdets,
    ∇ind,
    imp_mean,
    meandiff,
    imp_inv,
    mult,
    commutation_indices,
    nothing,

    zeros(parameter_type, 1),
    zeros(parameter_type, n_par),
    zeros(parameter_type, n_par, n_par)
    )
end

############################################################################
### functors
############################################################################

function (semfiml::SemFIML)(par, F, G, H, model::Sem{O, I, L, D}) where {O, I <: SemImplySymbolic, L, D}

    if H throw(DomainError(H, "hessian for FIML is not implemented (yet)")) end

    if !check_fiml(semfiml, model)
        if G semfiml.gradient .+= 1.0 end
        if F semfiml.objective[1] = Inf end
    else
        copy_per_pattern!(semfiml, model)
        batch_cholesky!(semfiml, model)
        #batch_sym_inv_update!(semfiml, model)
        batch_inv!(semfiml, model)
        for i in 1:size(pattern_n_obs(observed(model)), 1)
            semfiml.meandiff[i] .= obs_mean(observed(model))[i] - semfiml.imp_mean[i]
        end
        #semfiml.logdets .= -logdet.(semfiml.inverses)

        if G
            ∇F_FIML(semfiml.gradient, rows(observed(model)), semfiml, model)
            semfiml.gradient .= semfiml.gradient/n_obs(observed(model))
        end

        if F
            F_FIML(semfiml.objective, rows(observed(model)), semfiml, model)
            semfiml.objective[1] = semfiml.objective[1]/n_obs(observed(model))
        end

    end
    
end

function (semfiml::SemFIML)(par, F, G, H, model::Sem{O, I, L, D}) where {O, I <: RAM, L, D}

    if H throw(DomainError(H, "hessian for FIML is not implemented (yet)")) end

    if !check_fiml(semfiml, model)
        if G semfiml.gradient .+= 1.0 end
        if F semfiml.objective[1] = Inf end
    else
        copy_per_pattern!(semfiml, model)
        batch_cholesky!(semfiml, model)
        #batch_sym_inv_update!(semfiml, model)
        batch_inv!(semfiml, model)
        for i in 1:size(pattern_n_obs(observed(model)), 1)
            semfiml.meandiff[i] .= obs_mean(observed(model))[i] - semfiml.imp_mean[i]
        end
        #semfiml.logdets .= -logdet.(semfiml.inverses)

        if G
            ∇F_FIML(semfiml.gradient, rows(observed(model)), semfiml, model)
            semfiml.gradient .= semfiml.gradient/n_obs(observed(model))
        end

        if F
            F_FIML(semfiml.objective, rows(observed(model)), semfiml, model)
            semfiml.objective[1] = semfiml.objective[1]/n_obs(observed(model))
        end

    end
    
end

############################################################################
### Recommended methods
############################################################################

objective(lossfun::SemFIML) = lossfun.objective
gradient(lossfun::SemFIML) = lossfun.gradient
hessian(lossfun::SemFIML) = lossfun.hessian

update_observed(lossfun::SemFIML, observed::SemObs; kwargs...) = SemFIML(;observed = observed, kwargs...)

############################################################################
### additional functions
############################################################################

function F_one_pattern(meandiff, inverse, obs_cov, logdet, N)
    F = logdet
    F += meandiff'*inverse*meandiff
    if N > one(N)
        F += dot(obs_cov, inverse)
    end
    F = N*F
    return F
end

function ∇F_one_pattern(μ_diff, Σ⁻¹, S, pattern, ∇ind, N, Jμ, JΣ, model)
    diff⨉inv = μ_diff'*Σ⁻¹

    if N > one(N)
        JΣ[∇ind] .+= N*vec(Σ⁻¹*(I - S*Σ⁻¹ - μ_diff*diff⨉inv))
        @. Jμ[pattern] += (N*2*diff⨉inv)'

    else
        JΣ[∇ind] .+= vec(Σ⁻¹*(I - μ_diff*diff⨉inv))
        @. Jμ[pattern] += (2*diff⨉inv)'
    end

end

function ∇F_fiml_outer(JΣ, Jμ, imply::SemImplySymbolic, model, semfiml)
    G = transpose(JΣ'*∇Σ(imply)-Jμ'*∇μ(imply))
    return G
end

function ∇F_fiml_outer(JΣ, Jμ, imply, model, semfiml)

    Iₙ = sparse(1.0I, size(A(imply))...)
    P = kron(F⨉I_A⁻¹(imply), F⨉I_A⁻¹(imply))
    Q = kron(S(imply)*I_A(imply)', Iₙ)
    #commutation_matrix_pre_square_add!(Q, Q)
    Q2 = commutation_matrix_pre_square(Q, semfiml.commutation_indices)

    ∇Σ = P*(∇S(imply) + (Q+Q2)*∇A(imply))

    ∇μ = F⨉I_A⁻¹(imply)*∇M(imply) + kron((I_A(imply)*M(imply))', F⨉I_A⁻¹(imply))*∇A(imply)

    G = transpose(JΣ'*∇Σ-Jμ'*∇μ)

    return G
end

function F_FIML(F, rows, semfiml, model)
    F[1] = zero(eltype(F))
    for i = 1:size(rows, 1)
        F[1] += F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            obs_cov(observed(model))[i], 
            semfiml.logdets[i], 
            pattern_n_obs(observed(model))[i])
    end
end

function ∇F_FIML(G, rows, semfiml, model)
    Jμ = zeros(Int64(n_man(model)))
    JΣ = zeros(Int64(n_man(model)^2))
    
    for i = 1:size(rows, 1)
        ∇F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            obs_cov(observed(model))[i], 
            patterns(observed(model))[i],
            semfiml.∇ind[i],
            pattern_n_obs(observed(model))[i],
            Jμ,
            JΣ,
            model)
    end
    G .= ∇F_fiml_outer(JΣ, Jμ, imply(model), model, semfiml)
end

function copy_per_pattern!(inverses, source_inverses, means, source_means, patterns)
    @views for i = 1:size(patterns, 1)
        inverses[i] .=
            source_inverses[
                patterns[i],
                patterns[i]]
    end

    @views for i = 1:size(patterns, 1)
        means[i] .=
            source_means[patterns[i]]
    end
end

#= function copy_per_pattern!(inverses, source_inverses, means, source_means, patterns, which_source_pattern)
    @views for i = 1:size(patterns, 1)
        inverses[i] .=
            source_inverses[which_source_pattern[i]][
                patterns[i],
                patterns[i]]
    end

    @views for i = 1:size(patterns, 1)
        means[i] .=
            source_means[which_source_pattern[i]][patterns[i]]
    end
end =#

copy_per_pattern!(
    semfiml, 
    model::M where {M <: AbstractSem}) = 
    copy_per_pattern!(
        semfiml.inverses, 
        Σ(imply(model)), 
        semfiml.imp_mean, 
        μ(imply(model)), 
        patterns(observed(model)))

#= copy_per_pattern!(semfiml, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L , I <: ImplyDefinition, D} = 
    copy_per_pattern!(
        semfiml.inverses, 
        imply(model).imp_cov, 
        semfiml.imp_mean, 
        imply(model).imp_mean, 
        semfiml.interaction.missing_patterns,
        semfiml.interaction.gradientroup_imp_per_comb) =#


function batch_cholesky!(semfiml, model)
    for i = 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Symmetric(semfiml.inverses[i]))
        semfiml.logdets[i] = logdet(semfiml.choleskys[i])
    end
    return true
end

#= function batch_cholesky!(semfiml, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L, I <: ImplyDefinition, D}
    for i = 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Symmetric(semfiml.inverses[i]); check = false)
        if !isposdef(semfiml.choleskys[i]) return false end
    end
    return true
end =#

function check_fiml(semfiml, model)
    copyto!(semfiml.imp_inv, Σ(imply(model)))
    a = cholesky!(Symmetric(semfiml.imp_inv); check = false)
    return isposdef(a)
end

get_n_nodes(specification::RAMMatrices) = specification.size_F[2]
get_n_nodes(specification::ParameterTable) = length(specification.observed_vars) + length(specification.latent_vars)

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemFIML)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end
