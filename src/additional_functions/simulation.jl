# convenience functions for simulation studies

#####################################################################################################
# change observed (data) without reconstructing the whole model
#####################################################################################################

# use the same observed type as before
swap_observed(model::Sem; kwargs...) = swap_observed(model, typeof(model.observed).name.wrapper; kwargs...)

# construct a new observed type
swap_observed(model::Sem, observed_type::DataType; kwargs...) = swap_observed(model, observed_type(kwargs...))


function swap_observed(model::Sem, observed::SemObs)
    return Sem(
        new_observed, 
        update_observed(model.imply, new_observed; kwargs...),
        update_observed(model.loss, new_observed; kwargs...),
        update_observed(model.diff, new_observed; kwargs...)
        )
end

function update_observed(imply::SemImpy, observed::SemObs) end

function update_observed(lossfun::SemLossFunction, observed::SemObs) end

function update_observed(diff::SemDiff, observed::SemObs) end