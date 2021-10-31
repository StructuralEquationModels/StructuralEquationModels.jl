############################################################################
### Types
############################################################################

struct SemFIML{INV, C, L, O, M, IM, I, T, U, V, W} <: SemLossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    ∇ind::O
    imp_mean::IM
    meandiff::M
    imp_inv::I
    mult::T
    objective::U
    grad::V
    interaction::W
end

############################################################################
### Constructors
############################################################################

function SemFIML(observed::O where {O <: SemObs}, objective, grad)

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
    copy(objective),  
    copy(grad),
    nothing
    )
end

############################################################################
### functors
############################################################################

function (semfiml::SemFIML)(par, F, G, H, model, weight = nothing)

    if !isnothing(H) stop("hessian for ML is not implemented (yet)") end

    if !check_fiml(semfiml, model)
        if !isnothing(G) G .+= 1.0 end
        if !isnothing(F) return Inf end
    end

    copy_per_pattern!(semfiml, model)
    batch_cholesky!(semfiml, model)
    #batch_sym_inv_update!(semfiml, model)
    batch_inv!(semfiml, model)
    for i in 1:size(model.observed.pattern_n_obs, 1)
        @. semfiml.meandiff[i] = model.observed.obs_mean[i] - semfiml.imp_mean[i]
    end
    #semfiml.logdets .= -logdet.(semfiml.inverses)

    if !isnothing(G)
        ∇F_FIML(semfiml.grad, model.observed.rows, semfiml, model)
        if !isnothing(weight)
            @. semfiml.grad = weight*semfiml.grad
        end
        G .+= semfiml.grad
    end
    if !isnothing(F)
        F = F_FIML(zero(eltype(par)), model.observed.rows, semfiml, model)
        if !isnothing(weight)
            F = weight*F
        end
        return F
    end
    F = zero(eltype(par))
    F = F_FIML(F, model.observed.rows, semfiml, model)
    return F
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

function ∇F_one_pattern(μ_diff, Σ⁻¹, S, pattern, ∇ind, N, model)
    diff⨉inv = μ_diff'*Σ⁻¹

    if N > one(N)
        in1 = Σ⁻¹*(I - S*Σ⁻¹ - μ_diff*diff⨉inv)
        grad = vec(in1)'*model.imply.∇Σ[∇ind, :]
        grad -= 2*diff⨉inv*model.imply.∇μ[pattern, :]
        grad = N*grad
    else
        grad = 
        vec(
            Σ⁻¹*(I - μ_diff*diff⨉inv))'*model.imply.∇Σ[∇ind, :] -
            2*diff⨉inv*model.imply.∇μ[pattern, :]
    end
    return grad'
end

function F_FIML(F, rows, semfiml, model)
    for i = 1:size(rows, 1)
        F += F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            model.observed.obs_cov[i], 
            semfiml.logdets[i], 
            model.observed.pattern_n_obs[i])
    end
    return F
end

function ∇F_FIML(grad, rows, semfiml, model)
    grad .= 0.0
    for i = 1:size(rows, 1)
        grad += ∇F_one_pattern(
            semfiml.meandiff[i], 
            semfiml.inverses[i], 
            model.observed.obs_cov[i], 
            model.observed.patterns[i],
            semfiml.∇ind[i],
            model.observed.pattern_n_obs[i],
            model)
    end
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

function copy_per_pattern!(inverses, source_inverses, means, source_means, patterns, which_source_pattern)
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
end

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