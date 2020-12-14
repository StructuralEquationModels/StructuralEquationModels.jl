function (loss::Loss)(par, model)
    F = zero(eltype(par))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, model)
        # all functions have to have those arguments??
    end
    return F
end

# function (loss::Loss)(par, model, E, G)
#     if E != nothing
#         F = zero(eltype(model.imply.imp_cov))
#         for i = 1:length(loss.functions)
#             F += loss.functions[i](par, model, E, G)
#         end
#         return F
#     end
#     if G != nothing
#         G .= zero(eltype(G))
#         for i = 1:length(loss.functions)
#             loss.functions[i](par, model, E, G)
#         end
#         for i = 1:length(loss.functions)
#             G .+= loss.functions[i].grad
#         end
#
#     end
# end

function (loss::Loss)(par, model, E, G)

    if E != nothing

        for i = 1:length(loss.functions)
            loss.functions[i](par, model, E, G)
        end

        objective = zero(eltype(model.imply.imp_cov))
        for i = 1:length(loss.functions)
            objective += loss.functions[i].objective[1]
        end

        return objective
    end

    if G != nothing

        for i = 1:length(loss.functions)
            loss.functions[i](par, model, E, G)
        end

        G .= zero(eltype(G))
        for i = 1:length(loss.functions)
            G .+= loss.functions[i].grad
        end

    end
end


# function (loss::Loss)(par, model, E, G)
#
#     #common computations with 2 arguments
#     #store computations in fields
#     for i = 1:length(loss.functions)
#         loss.functions[i](par, model)
#     end
#
#     for i = 1:length(loss.functions)
#         loss.functions[i](par, model, E, G)
#     end
#
#     if E != nothing
#         objective = zero(eltype(model.imply.imp_cov))
#         for i = 1:length(loss.functions)
#             objective += loss.functions[i].objective[1]
#         end
#         return objective
#     end
#
#     if G != nothing
#
#         G .= zero(eltype(G))
#         for i = 1:length(loss.functions)
#             G .+= loss.functions[i].grad
#         end
#
#     end
# end

# function (lasso::SemLasso)(par, model)
# end
#
# function (lasso::SemLasso)(par, model, E, G) where {G <: Nothing}
#     lasso.F .= lasso.penalty*sum(transpose(par)[lasso.which])
#     return lasso.F[1]
# end
#
# function (lasso::SemLasso)(par, model, E, G) where {E <: Nothing}
#     model.imply()
#     ForwardDiff.gradient!(G, lasso(par, model), par)
# end

## Lossfunctions

struct SemML{
        I <: AbstractArray,
        T <: AbstractArray,
        U, V,
        W <: Union{Nothing, AbstractArray}} <: LossFunction
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
    meandiff::W
end

function SemML(observed::T, objective, grad) where {T <: SemObs}
    isnothing(observed.obs_mean) ?
        meandiff = nothing :
        meandiff = copy(observed.obs_mean)
    return SemML(
        copy(observed.obs_cov),
        copy(observed.obs_cov),
        copy(objective),
        copy(grad),
        meandiff
        )
end

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

# this catches everything that is not further optimized (including Duals)
function (semml::SemML)(par, model)
    if !isposdef(Hermitian(model.imply.imp_cov))
        F = Inf
    else
        F = logdet(model.imply.imp_cov) +
             tr(inv(model.imply.imp_cov)*model.observed.obs_cov)
    end
    return F
end

# maximum speed for finite differences
function (semml::SemML)(par, model::Sem{O, I, L, D}) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}
    semml.imp_inv .= model.imply.imp_cov
    a = cholesky!(Hermitian(semml.imp_inv); check = false)
    if !isposdef(a)
        F = Inf
    else
        ld = logdet(a)
        semml.imp_inv .= LinearAlgebra.inv!(a)
        #inv_cov = inv(a)
        mul!(semml.mult, semml.imp_inv, model.observed.obs_cov)
        if !isnothing(model.imply.imp_mean)
            @. semml.meandiff = model.observed.obs_mean - model.imply.imp_mean
            F_mean = semml.meandiff'*semml.imp_inv*semml.meandiff
        else
            F_mean = zero(eltype(par))
        end
        #mul!()
        F = ld +
            tr(semml.mult) + F_mean
    end
    return F
end

function (semml::SemML)(par, model::Sem{O, I, L, D}, E, G) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}

    a = cholesky(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        if E != nothing
            semml.objective .= Inf
        end
        if G != nothing
            semml.grad .= 0.0
        end
    else
        ld = logdet(a)
        inv_cov = inv(a)
        if E != nothing
            mul!(semml.mult, inv_cov, model.observed.obs_cov)
            semml.objective .= ld + tr(semml.mult)
        end
        if G != nothing
            model.diff.B!(model.diff.B, par) # B = inv(I-A)
            model.diff.E!(model.diff.E, par) # E = B*S*B'
            let B = model.diff.B, E = model.diff.E,
                Σ_inv = inv_cov, F = model.diff.F,
                D = model.observed.obs_cov
                for i = 1:size(par, 1)
                    S_der = sparse(model.diff.S_ind_vec[i]..., model.diff.matsize...)
                    A_der = sparse(model.diff.A_ind_vec[i]..., model.diff.matsize...)

                    term = F*B*A_der*E*F'
                    Σ_der = Array(F*B*S_der*B'F' + term + term')

                    semml.grad[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
                end
            end
        end
    end
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

### regularized
# those do not need to dispatch I guess
struct SemLasso{P, W} <: LossFunction
    penalty::P
    which::W
end

function (lasso::SemLasso)(par, model)
      F = lasso.penalty*sum(transpose(par)[lasso.which])
end

struct SemRidge{P, W} <: LossFunction
    penalty::P
    which::W
end

function (ridge::SemRidge)(par, implied, observed)
      F = ridge.penalty*sum(transpose(par)[ridge.which].^2)
end
