# function Sem(;
#         semobs::O,
#         imply = nothing,
#         loss = nothing,
#         semdiff = nothing,
#         algorithm = nothing
#         ) where {
#             O <: SemObs
#         }
#     if isnothing(imply) imply = ImplyCommon()
# end



### Two versions. The second one could be easier to construct
function (model::Sem)(par)
    model.imply(par)
    F = model.loss(
        par,
        model)
    return(F)
end

function computeloss(model, par)
    model.imply(par)
    F = model.loss(par, model)
    return(F)
end
