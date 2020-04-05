import StatsBase.fit
function fit(model::model)
    optimize(
    par -> model.objective(par, model),
    model.par,
    model.optimizer,
    autodiff = :forward)
end
