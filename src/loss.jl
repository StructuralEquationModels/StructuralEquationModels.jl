function (loss::Loss)(par, model)
    F = zero(eltype(model.imply.imp_cov))
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

struct SemML{I <: AbstractArray, T <: AbstractArray, U, V} <: LossFunction
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
end

function SemML(observed::T, objective, grad) where {T <: SemObs}
    return SemML(
        copy(observed.obs_cov),
        copy(observed.obs_cov),
        copy(objective),
        copy(grad)
        )
end

struct SemFIML <: LossFunction
    #here is space for preallocations
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
    a = cholesky!(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        F = Inf
    else
        ld = logdet(a)
        #model.imply.imp_cov .= LinearAlgebra.inv!(a)
        inv_cov = inv(a)
        mul!(semml.mult, inv_cov, model.observed.obs_cov)
        #mul!()
        F = ld +
            tr(semml.mult)
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


struct teststruc2
    A::Array{Float64, 1}
end

mystruc = teststruc2([4.0])

mystruc.A .= [5.0]
