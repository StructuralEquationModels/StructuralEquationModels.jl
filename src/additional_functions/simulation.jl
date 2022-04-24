# convenience functions for simulation studies

#####################################################################################################
# change observed (data) without reconstructing the whole model
#####################################################################################################

# use the same observed type as before
swap_observed(model::AbstractSemSingle; kwargs...) = swap_observed(model, typeof(observed(model)).name.wrapper; kwargs...)

# construct a new observed type
swap_observed(model::AbstractSemSingle, observed_type; kwargs...) = swap_observed(model, observed_type(;kwargs...); kwargs...)

swap_observed(model::AbstractSemSingle, new_observed::SemObs; kwargs...) =
    swap_observed(model, observed(model), imply(model), loss(model), diff(model), new_observed; kwargs...)

function swap_observed(
        model::AbstractSemSingle, 
        old_observed,
        imply,
        loss,
        diff,
        new_observed::SemObs; 
        kwargs...)

    kwargs = Dict{Symbol, Any}(kwargs...)
    
    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(old_observed)
    kwargs[:imply_type] = typeof(imply)
    kwargs[:loss_types] = [typeof(lossfun) for lossfun in loss.functions]
    kwargs[:diff_type] = typeof(diff)

    # update imply
    imply = update_observed(imply, new_observed; kwargs...)
    kwargs[:imply] = imply
    kwargs[:n_par] = n_par(imply)

    # update loss
    loss = update_observed(loss, new_observed; kwargs...)
    kwargs[:loss] = loss

    # update diff
    diff = update_observed(diff, new_observed; kwargs...)

    #new_imply = update_observed(model.imply, new_observed; kwargs...)

    return Sem(
        new_observed, 
        update_observed(model.imply, new_observed; kwargs...),
        update_observed(model.loss, new_observed; kwargs...),
        update_observed(model.diff, new_observed; kwargs...)
        )
end

function update_observed(loss::SemLoss, new_observed; kwargs...)
    new_functions = Tuple(update_observed(lossfun, new_observed; kwargs...) for lossfun in loss.functions)
    return SemLoss(
        new_functions,
        loss.weights,
        loss.objective,
        loss.gradient,
        loss.hessian)
end