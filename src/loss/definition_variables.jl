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
    batch_inv!(semml, model)
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

function findrow(r, rows)
    for i in 1:length(rows)
        if r ∈ rows[i]
            return i
        end
    end
end
