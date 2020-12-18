# Full Information Maximum Likelihood Estimation

struct SemFIML{
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        IM <: AbstractArray,
        I <: AbstractArray,
        T <: AbstractArray,
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    imp_mean::IM
    meandiff::M
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
end

function SemFIML(observed::O where {O <: SemObs}, objective, grad)

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
    copy(grad)
    )
    end


    function (semfiml::SemFIML)(par, model::Sem{O, I, L, D}) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}

    if isnothing(model.imply.imp_mean) 
        error("A model implied meanstructure is needed for FIML")
    end

    copyto!(semfiml.imp_inv, model.imply.imp_cov)
    a = cholesky!(Hermitian(semfiml.imp_inv); check = false)

    if !isposdef(a)
        F = Inf
    else
        @views for i = 1:size(semfiml.inverses, 1)
            semfiml.inverses[i] .=
                model.imply.imp_cov[
                    model.observed.patterns[i],
                    model.observed.patterns[i]]
        end

        @views for i = 1:size(semfiml.imp_mean, 1)
            semfiml.imp_mean[i] .=
                model.imply.imp_mean[model.observed.patterns[i]]
        end

        for i = 1:size(semfiml.inverses, 1)
            semfiml.choleskys[i] = cholesky!(Hermitian(semfiml.inverses[i]))
        end

        #ld = logdet(a)
        semfiml.logdets .= logdet.(semfiml.choleskys)

        #semml.imp_inv .= LinearAlgebra.inv!(a)
        for i = 1:size(semfiml.inverses, 1)
            semfiml.inverses[i] .= LinearAlgebra.inv!(semfiml.choleskys[i])
        end

        F = zero(eltype(par))
        

        for i = 1:size(model.observed.rows, 1)
            for j in model.observed.rows[i]
                let (imp_mean, meandiff, inverse, data, logdet) =
                    (semfiml.imp_mean[i],
                    semfiml.meandiff[i],
                    semfiml.inverses[i],
                    model.observed.data_perperson[j],
                    semfiml.logdets[i])

                    F += F_one_person(imp_mean, meandiff, inverse, data, logdet)

                end
            end
        end
    end
    return F
end