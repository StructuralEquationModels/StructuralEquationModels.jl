abstract type LossFunction <: Function end

struct SemML{T <: AbstractArray} <: LossFunction
    mult::T # is this type known?
end

function SemML(observed::T) where {T <: SemObs}
    return SemML(copy(observed.obs_cov)) # what should this type be?
end

struct SemFIML <: LossFunction
    #here is space for preallocations
end

# this catches everything that is not further optimized (including Duals)
function (semml::SemML)(
    par,
    model
    )
    if !isposdef(model.imply.imp_cov)
        F = Inf
    else
        a = cholesky!(model.imply.imp_cov)
        ld = logdet(a)
        model.imply.imp_cov .= LinearAlgebra.inv!(a)
        mul!(semml.mult, model.observed.obs_cov, model.imply.imp_cov)
        #mul!()
        F = ld +
            tr(semml.mult)
    end
    return F
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



### wrapper type that applies all specified loss functions and sums those
struct Loss{F <: Array{T} where {T <: LossFunction}}
    functions::F
end

function (loss::Loss)(par, model)
    F = zero(eltype(model.imply.imp_cov))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, model)
        # all functions have to have those arguments??
    end
    return F
end
