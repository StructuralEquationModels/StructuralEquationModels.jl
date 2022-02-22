############################################################################
### Types
############################################################################

struct SemFIML{INV, C, L, O, M, IM, I, T, W, FT, GT, HT} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    ∇ind::O
    imp_mean::IM
    meandiff::M
    imp_inv::I
    mult::T
    interaction::W

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemFIML(;observed, n_par, parameter_type = Float64, kwargs...)

    inverses = broadcast(x -> zeros(x, x), Int64.(observed.pattern_nvar_obs))
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))

    n_patterns = size(observed.rows, 1)
    logdets = zeros(n_patterns)

    imp_mean = zeros.(Int64.(observed.pattern_nvar_obs))
    meandiff = zeros.(Int64.(observed.pattern_nvar_obs))

    imp_inv = zeros(size(observed.data, 2), size(observed.data, 2))
    mult = similar.(inverses)

    n_man = Int64(observed.n_man)
    ∇ind = vec(CartesianIndices(Array{Float64}(undef, n_man, n_man)))
    ∇ind = [findall(x -> !(x[1] ∈ ind || x[2] ∈ ind), ∇ind) for ind in observed.patterns_not]

    return SemFIML(
    inverses,
    choleskys,
    logdets,
    ∇ind,
    imp_mean,
    meandiff,
    imp_inv,
    mult,
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

    if H stop("hessian for FIML is not implemented (yet)") end

    if !check_fiml(semfiml, model)
        if G semfiml.G .+= 1.0 end
        if F semfiml.F[1] = Inf end
    else
        copy_per_pattern!(semfiml, model)
        batch_cholesky!(semfiml, model)
        #batch_sym_inv_update!(semfiml, model)
        batch_inv!(semfiml, model)
        for i in 1:size(model.observed.pattern_n_obs, 1)
            @. semfiml.meandiff[i] = model.observed.obs_mean[i] - semfiml.imp_mean[i]
        end
        #semfiml.logdets .= -logdet.(semfiml.inverses)

        if G
            ∇F_FIML(semfiml.G, model.observed.rows, semfiml, model)
            @. semfiml.G = semfiml.G/model.observed.n_obs
        end

        if F
            F_FIML(semfiml.F, model.observed.rows, semfiml, model)
            semfiml.F[1] = semfiml.F[1]/model.observed.n_obs
        end

    end
    
end

function (semfiml::SemFIML)(par, F, G, H, model::Sem{O, I, L, D}) where {O, I <: RAM, L, D}

    if H stop("hessian for FIML is not implemented (yet)") end

    if !check_fiml(semfiml, model)
        if G semfiml.G .+= 1.0 end
        if F semfiml.F[1] = Inf end
    else
        copy_per_pattern!(semfiml, model)
        batch_cholesky!(semfiml, model)
        #batch_sym_inv_update!(semfiml, model)
        batch_inv!(semfiml, model)
        for i in 1:size(model.observed.pattern_n_obs, 1)
            @. semfiml.meandiff[i] = model.observed.obs_mean[i] - semfiml.imp_mean[i]
        end
        #semfiml.logdets .= -logdet.(semfiml.inverses)

        if G
            ∇F_FIML(semfiml.G, model.observed.rows, semfiml, model)
            @. semfiml.G = semfiml.G/model.observed.n_obs
        end

        if F
            F_FIML(semfiml.F, model.observed.rows, semfiml, model)
            semfiml.F[1] = semfiml.F[1]/model.observed.n_obs
        end

    end
    
end

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

function ∇F_fiml_outer(JΣ, Jμ, imply::SemImplySymbolic, model)
    G = transpose(JΣ'*imply.∇Σ-Jμ'*imply.∇μ)
    return G
end

function ∇F_fiml_outer(JΣ, Jμ, imply, model)

    Iₙ = sparse(1.0I, size(imply.A)...)
    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Q = kron(imply.S*imply.I_A', Iₙ)
    commutation_matrix_pre_square_add!(Q, Q)

    ∇Σ = P*(imply.∇S + Q*imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹*imply.∇M + kron((imply.I_A*imply.M)', imply.F⨉I_A⁻¹)*imply.∇A

    G = transpose(JΣ'*∇Σ-Jμ'*∇μ)

    return G
end

function F_FIML(F, rows, semfiml, model)
    F[1] = zero(eltype(F))
    for i = 1:size(rows, 1)
        F[1] += F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            model.observed.obs_cov[i], 
            semfiml.logdets[i], 
            model.observed.pattern_n_obs[i])
    end
end

function ∇F_FIML(G, rows, semfiml, model)
    Jμ = zeros(Int64(model.observed.n_man))
    JΣ = zeros(Int64(model.observed.n_man^2))
    
    for i = 1:size(rows, 1)
        ∇F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            model.observed.obs_cov[i], 
            model.observed.patterns[i],
            semfiml.∇ind[i],
            model.observed.pattern_n_obs[i],
            Jμ,
            JΣ,
            model)
    end
    G .= ∇F_fiml_outer(JΣ, Jμ, model.imply, model)
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
        model.imply.Σ, 
        semfiml.imp_mean, 
        model.imply.μ, 
        model.observed.patterns)

#= copy_per_pattern!(semfiml, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L , I <: ImplyDefinition, D} = 
    copy_per_pattern!(
        semfiml.inverses, 
        model.imply.imp_cov, 
        semfiml.imp_mean, 
        model.imply.imp_mean, 
        semfiml.interaction.missing_patterns,
        semfiml.interaction.group_imp_per_comb) =#


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
    copyto!(semfiml.imp_inv, model.imply.Σ)
    a = cholesky!(Symmetric(semfiml.imp_inv); check = false)
    return isposdef(a)
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemFIML)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end