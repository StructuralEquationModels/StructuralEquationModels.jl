##################### compute loss per pattern ############################

# type
struct SemFIML{
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        IM <: AbstractArray,
        I <: AbstractArray,
        T <: AbstractArray,
        U,
        V,
        W} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    imp_mean::IM
    meandiff::M
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
    interaction::W
end

function SemFIML(observed::O where {O <: SemObs}, 
    imply::I where {I <: Imply}, objective, grad)

    if isnothing(imply.imp_mean) 
        error("A model implied meanstructure is needed for FIML")
    end

    inverses = broadcast(x -> zeros(x, x), Int64.(observed.pattern_nvar_obs))
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))

    n_patterns = size(observed.rows, 1)
    logdets = zeros(n_patterns)

    imp_mean = zeros.(Int64.(observed.pattern_nvar_obs))
    meandiff = zeros.(Int64.(observed.pattern_nvar_obs))

    imp_inv = zeros(size(observed.data, 2), size(observed.data, 2))
    mult = similar.(inverses)

    return SemFIML(
    inverses,
    choleskys,
    logdets,
    imp_mean,
    meandiff,
    imp_inv,
    mult,
    copy(objective),  
    copy(grad),
    nothing
    )
end

function F_one_pattern(imp_mean, meandiff, inverse, obs_cov, obs_mean, logdet, N)
    F = logdet
    @. meandiff = obs_mean - imp_mean
    F += dot(meandiff, inverse, meandiff)
    if N > one(N)
        F += tr(obs_cov*inverse)
    end
    F = N*F
    return F
end

function F_FIML(F, rows, semfiml, model)
    for i = 1:size(rows, 1)
        let (imp_mean, meandiff, inverse, obs_cov, obs_mean, logdet, N) =
            (semfiml.imp_mean[i],
            semfiml.meandiff[i],
            semfiml.inverses[i],
            model.observed.obs_cov[i],
            model.observed.obs_mean[i],
            semfiml.logdets[i],
            model.observed.pattern_n_obs[i])

            F += F_one_pattern(imp_mean, meandiff, inverse, obs_cov, obs_mean, logdet, N)
        end
    end
    return F
end

function (semfiml::SemFIML)(
    par, 
    model::Sem{O, I, L, D}
    ) where 
    {O <: SemObs, 
    L <: Loss, 
    I <: Imply, 
    D <: SemDiff}

    if !check_fiml(semfiml, model) return Inf end

    copy_per_pattern!(semfiml, model)
    batch_cholesky!(semfiml, model)
    semfiml.logdets .= logdet.(semfiml.choleskys)
    batch_inv!(semfiml, model)

    F = zero(eltype(par))
    F = F_FIML(F, model.observed.rows, semfiml, model)
    return F
end

# Helper functions
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
    semfiml::SemFIML, 
    model::M where {M <: AbstractSem}) = 
    copy_per_pattern!(
        semfiml.inverses, 
        model.imply.imp_cov, 
        semfiml.imp_mean, 
        model.imply.imp_mean, 
        model.observed.patterns)

copy_per_pattern!(semfiml::SemFIML, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L , I <: ImplyDefinition, D} = 
    copy_per_pattern!(
        semfiml.inverses, 
        model.imply.imp_cov, 
        semfiml.imp_mean, 
        model.imply.imp_mean, 
        semfiml.interaction.missing_patterns,
        semfiml.interaction.group_imp_per_comb)


function batch_cholesky!(semfiml::SemFIML, model)
    for i = 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Hermitian(semfiml.inverses[i]))
    end
    return true
end

function batch_cholesky!(semfiml::SemFIML, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L, I <: ImplyDefinition, D}
    for i = 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Hermitian(semfiml.inverses[i]); check = false)
        if !isposdef(semfiml.choleskys[i]) return false end
    end
    return true
end

function check_fiml(semfiml, model)
    copyto!(semfiml.imp_inv, model.imply.imp_cov)
    a = cholesky!(Hermitian(semfiml.imp_inv); check = false)
    return isposdef(a)
end

##################### definition variables ############################
# interaction
struct SemInteractionFIMLDefition{A, B, C, D} <: SemInteraction
    rows_per_comb::A
    group_obs_per_comb::B
    group_imp_per_comb::C
    missing_patterns::D
end

# constructor 
function SemFIML(
        observed::O where {O <: SemObsMissing}, 
        imply::I where {I <: ImplySymbolicDefinition}, 
        objective,
        grad) 

        
        group_obs = findrow.(1:Int64(observed.n_obs), [observed.rows])
        group_imp = findrow.(1:Int64(observed.n_obs), [imply.rows])
        
        rows_per_comb = Vector{Vector{Int64}}()
        group_obs_per_comb = Vector{Int64}()
        group_imp_per_comb = Vector{Int64}()
        
        for i in 1:Int64(observed.n_obs)
            new = true
            for j in 1:length(rows_per_comb)
                if (group_obs[i] == group_obs_per_comb[j]) && (group_imp[i] == group_imp_per_comb[j])
                    push!(rows_per_comb[j], i)
                    new = false
                end
            end
            if new
                push!(rows_per_comb, [i])
                push!(group_obs_per_comb, group_obs[i])
                push!(group_imp_per_comb, group_imp[i])
            end
        end
        
        sorted_ind = sortperm(collect(zip(group_imp_per_comb, group_obs_per_comb)))
        rows_per_comb = rows_per_comb[sorted_ind]
        group_obs_per_comb = group_obs_per_comb[sorted_ind]
        group_imp_per_comb = group_imp_per_comb[sorted_ind]
        
        # inverses
        nvar_obs_per_comb = Int64.(observed.pattern_nvar_obs[group_obs_per_comb])
        inverses = zeros.(nvar_obs_per_comb, nvar_obs_per_comb)
        
        # choleskys
        choleskys = Vector{Cholesky{Float64, Matrix{Float64}}}(undef, length(inverses))
        
        n_patterns = size(group_obs_per_comb, 1)
        
        logdets = zeros(size(inverses, 1))
        
        imp_mean = zeros.(size.(inverses, 1))
        meandiff = zeros.(size.(inverses, 1))
        
        imp_inv = similar(imply.imp_cov)
        mult = similar.(inverses)
        
        # missing patterns
        missing_patterns = observed.patterns[group_obs_per_comb]

    return SemFIML(
    inverses,
    choleskys,
    logdets,
    imp_mean,
    meandiff,
    imp_inv,
    mult,
    copy(objective),
    copy(grad),
    SemInteractionFIMLDefition(rows_per_comb, group_obs_per_comb, group_imp_per_comb, missing_patterns)
    )
end

# loss function for defvars + fiml
function (semfiml::SemFIML)(par, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L <: Loss, I <: ImplyDefinition, D <: SemFiniteDiff}
    
    copy_per_pattern!(semfiml, model)
    if !batch_cholesky!(semfiml, model) return Inf end
    semfiml.logdets .= logdet.(semfiml.choleskys)
    batch_inv!(semfiml, model)

    F = zero(eltype(par))
    F = F_FIML(F, semfiml.interaction.rows_per_comb, semfiml, model)
    return F
end