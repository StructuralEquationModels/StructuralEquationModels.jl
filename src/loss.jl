abstract type LossFunction <: Function end

struct SemML <: LossFunction
    #here is space for preallocations
end

# this catches everything that is not further optimized (including Duals)
function (semml::SemML)(par, implied, observed)
      F =
      return F
end

# this is for Floats, so for example with finite differences
# AbstractFloat, because Float32 remains an option
function (semml::SemML)(par, implied::Array{T}, observed) where {T <: AbstractFloat}
      F =
      return F
end

# sparse
function (semml::SemML)(par, implied::SomethingSparse, observed)
      F =
      return F
end

### regularized
# those do no need to dispatch I guess
struct SemLasso{P, W} <: LossFunction
    penalty::P
    which::W
end

function (lasso::SemLasso)(par, implied, observed)
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
struct Loss{F <: Array{LossFunction}}
    functions::F
end

function (loss::Loss)(par, implied, observed)
    F = zero(eltype(implied))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, implied, observed)
        # all functions have to have those arguments??
    end
    return F
end

a = [ForwardDiff.Dual(10) ForwardDiff.Dual(10)
    ForwardDiff.Dual(10) ForwardDiff.Dual(10)]
