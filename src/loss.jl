function (loss::Loss)(par, model)
    F = zero(eltype(model.imply.imp_cov))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, model)
        # all functions have to have those arguments??
    end
    return F
end

function (loss::Loss)(par, model, E, G)
    if E != nothing
        F = zero(eltype(model.imply.imp_cov))
        for i = 1:length(loss.functions)
            F += loss.functions[i](par, model, E, G)
        end
        return F
    end
    if G != nothing
        G .= 0
        for i = 1:length(loss.functions)
            loss.functions[i](par, model, E, G)
        end
        for i = 1:length(loss.functions)
            G += loss.functions[i].grad
        end

    end
end

## Lossfunctions

struct SemML{T <: AbstractArray, U} <: LossFunction
    mult::T # is this type known?
    grad::U
end

function SemML(observed::T) where {T <: SemObs}
    return SemML(copy(observed.obs_cov)) # what should this type be?
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
        model.imply.imp_cov .= LinearAlgebra.inv!(a)
        mul!(semml.mult, model.imply.imp_cov, model.observed.obs_cov)
        #mul!()
        F = ld +
            tr(semml.mult)
    end
    return F
end

function (semml::SemML)(par, model::Sem{O, I, L, D}, E, G) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    a = cholesky!(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        if E != nothing
            return F = Inf
        end
        if G != nothing
            semml.grad .= 0.0
        end
    else
        model.imply.imp_cov .= LinearAlgebra.inv!(a)
        if E != nothing
            ld = logdet(a)
            mul!(semml.mult, model.imply.imp_cov, model.observed.obs_cov)
            #mul!()
            F = ld +
                tr(semml.mult)
            return F
        end
        if G != nothing
            model.diff.B!(model.diff.B, par) # B = inv(I-A)
            model.diff.E!(model.diff.E, par) # E = B*S*B'
            let B = model.diff.B, E = model.diff.E,
                Σ_inv = model.imply.imp_cov, F = model.diff.F,
                D = model.observed.obs_cov
                for i = 1:size(par, 1)
                    S_der = sparse(model.diff.S_ind_vec[i]..., model.diff.matsize...)
                    A_der = sparse(model.diff.A_ind_vec[i]..., model.diff.matsize...)

                    term = F*B*A_der*E*F'
                    Σ_der =  F*B*S_der*B'F' + term + term'

                    semml.grad[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
                end
            end
        end
    end
end

# this is for Floats, so for example with finite differences
# AbstractFloat, because Float32 remains an option
function (semml::SemML)(
    par,
    implied::A,
    observed
    ) where {
    A <: Array{AbstractFloat}
    }
      F = 0.0
      return F
end



# sparse
function (semml::SemML)(par, implied::SparseMatrixCSC, observed)
      F = 0.0
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
