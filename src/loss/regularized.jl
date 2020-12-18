
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