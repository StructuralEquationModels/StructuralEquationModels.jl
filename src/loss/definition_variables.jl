# move data_rowwise to observed
# 

#= struct SemDefinition{ #################### call it per person or sth????
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    meandiff::M
    #data_rowwise::DP  -> add to semobs
    objective::U
    grad::V
end =#

struct SemDefinition{ #################### call it per person or sth????
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        I <: AbstractArray,
        IM <: Union{AbstractArray, Nothing},
        T <: AbstractArray,
        DP <: AbstractArray,
        K <: Union{AbstractArray, Nothing},
        R <: Union{AbstractArray, Nothing},
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    meandiff::M
    imp_inv::I
    imp_mean::IM
    mult::T
    data_perperson::DP
    keys::K
    rows::R
    objective::U
    grad::V
end

# constructor
function SemML(
        observed::O where {O <: SemObs}, 
        imply::I where {I <: ImplyDefinition}, 
        objective,
        grad) 
    n_obs = Int64(observed.n_obs)
    n_man = Int64(observed.n_man)
    n_patterns = imply.n_patterns

    inverses = [zeros(n_man, n_man) for i = 1:n_patterns]
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, n_patterns)
    logdets = zeros(n_patterns)

    meandiff = [zeros(n_man) for i = 1:n_obs]

    return SemML(
    inverses,
    choleskys,
    nothing,
    logdets,
    meandiff,
    copy(objective),
    copy(grad)
    )
end

# loss
function (semml::SemML)(par, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: ImplyDefinition, D <: SemFiniteDiff}

    if !prepare_ML!(semml, model) return Inf end

    F = zero(eltype(par))
    F = F_Definition(F, semml, model)
    
    return F
end

function batch_cholesky!(semml::SemML, model)
    for i = 1:size(semml.choleskys, 1)
        semml.choleskys[i] = 
            cholesky!(Hermitian(model.imply.imp_cov[i]); check = false)
        if !isposdef(semml.choleskys[i]) return false end
    end
    return true
end

function prepare_ML!(semml, model)
    if !batch_cholesky!(semml, model) return false end
    semml.logdets .= logdet.(semml.choleskys)
    batch_inv!(semml)
    return true
end

function F_Definition(F, semml, model)
    for i = 1:Int64(model.imply.n_patterns)
        for j in model.imply.rows[i]
            let (imp_mean, meandiff, inverse, data, logdet) =
                (model.imply.imp_mean[i],
                semml.meandiff[i],
                semml.inverses[i],
                model.observed.data_rowwise[j],
                semml.logdets[i])

                F += F_one_person(imp_mean, meandiff, inverse, data, logdet)

            end
        end
    end
    return F
end


# constructor for defvars + fiml (observed <: SemObsMissing)
function SemDefinition(
        observed::O where {O <: SemObsMissing}, 
        imply::I where {I <: ImplySymbolicDefinition}, 
        objective,
        grad) 

        
    keys = findrow.(1:Int64(observed.n_obs), [observed.rows])

    rows_nested = Vector{Vector{Vector{Int64}}}()
    keys_inner_vec = Vector{Vector{Int64}}()

    for i in 1:length(imply.rows)
        nest = Vector{Vector{Int64}}()
        keys_inner = Vector{Int64}()
        for j in imply.rows[i]
            unknown = true
            key = keys[j]
            for k in 1:size(keys_inner, 1)
                if keys_inner[k] == key
                    push!(nest[k], j)
                    unknown = false
                end
            end
            if unknown
                push!(keys_inner, key)
                push!(nest, [j])
            end
        end
        push!(rows_nested, nest)
        push!(keys_inner_vec, keys_inner)
    end

    data_rowwise = [observed.data[i, :] for i = 1:Int64(observed.n_obs)]

    #rows_nested

    inverses = Vector{Vector{Array{Float64, 2}}}()
    for i = 1:size(keys_inner_vec, 1)
        inverses_defgroup = Vector{Array{Float64, 2}}()
        for j in keys_inner_vec[i] 
            nvar = Int64(observed.pattern_nvar_obs[j])
            push!(inverses_defgroup, zeros(nvar, nvar))
        end
        push!(inverses, inverses_defgroup)
    end

    #inverses

    #choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))
    choleskys = [
        Array{Cholesky{Float64,Array{Float64,2}},1}(
            undef, length(inverses[i])) for i in 1:length(inverses)]

    n_patterns = sum([size(inverses[i], 1) for i = 1:length(inverses)])
    logdets = [zeros(size(keys_inner_vec[i], 1)) for i = 1:size(keys_inner_vec, 1)]

    imp_mean = [zeros.(size.(inverses[i], 1)) for i = 1:size(inverses, 1)]
    meandiff = [zeros.(size.(inverses[i], 1)) for i = 1:size(inverses, 1)]

    imp_inv = similar(imply.imp_cov)
    mult = similar.(inverses)


    return SemDefinition(
    inverses,
    choleskys,
    logdets,
    meandiff,
    imp_inv,
    imp_mean,
    mult,
    data_rowwise,
    keys_inner_vec,
    rows_nested,
    copy(objective),
    copy(grad)
    )
end

#= # loss function for defvars + fiml
function (semdef::SemDefinition)(par, model::Sem{O, I, L, D}) where
    {O <: SemObsMissing, L <: Loss, I <: Imply, D <: SemFiniteDiff}

    if isnothing(model.imply.imp_mean) 
        error("A model implied meanstructure is needed for Definition Variables")
    end

    for i = 1:size(semdef.keys, 1)
        for j = 1:size(semdef.keys[i], 1)
            @views semdef.inverses[i][j] .=
                    model.imply.imp_cov[i][
                        model.observed.patterns[semdef.keys[i][j]],
                        model.observed.patterns[semdef.keys[i][j]]]
            semdef.choleskys[i][j] = 
                cholesky!(Hermitian(semdef.inverses[i][j]); check = false)
            if !isposdef(semdef.choleskys[i][j]) return Inf end
        end
    end

    @views  for i = 1:size(semdef.keys, 1)
                for j = 1:size(semdef.keys[i], 1)
                    semdef.imp_mean[i][j] .=
                        model.imply.imp_mean[i][
                            model.observed.patterns[semdef.keys[i][j]]]
                end
            end

    for i = 1:size(semdef.keys, 1)
        for j = 1:size(semdef.keys[i], 1)
            semdef.logdets[i][j] = logdet(semdef.choleskys[i][j])
        end
    end

    #semml.imp_inv .= LinearAlgebra.inv!(a)
    for i = 1:size(semdef.keys, 1)
        for j = 1:size(semdef.keys[i], 1)
            semdef.inverses[i][j] .= LinearAlgebra.inv!(semdef.choleskys[i][j])
        end
    end

    F = zero(eltype(par))

    for i = 1:size(semdef.rows, 1)
        for j = 1:size(semdef.rows[i], 1)
            for k = 1:size(semdef.rows[i][j], 1)
                let (imp_mean, meandiff, inverse, data, logdet) =
                    (semdef.imp_mean[i][j],
                    semdef.meandiff[i][j],
                    semdef.inverses[i][j],
                    model.observed.data_rowwise[semdef.rows[i][j][k]],
                    semdef.logdets[i][j])

                    F += F_one_person(imp_mean, meandiff, inverse, data, logdet)
                end
            end
        end
    end
    return F
end
 =#

function findrow(r, rows)
    for i in 1:length(rows)
        if r ∈ rows[i]
            return i
        end
    end
end
