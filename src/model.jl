struct Sem{L <: Loss, I <: Imply, D <: SemDiff}
    loss::L # list of loss functions
    imply::I # former ram
    algorithm # LBFGS(), Newton, etc... typed?
    diff::D
end


### Two versions. The second one could be easier to construct
function (model::Sem)(par)
    model.imply(par)
    F = model.loss(par, model.imply.implied, model.obs)
    return(F)
end

function computeloss(model, par)
    model.imply(par)
    F = model.loss(par, model.imply.implied, model.obs)
    return(F)
end
