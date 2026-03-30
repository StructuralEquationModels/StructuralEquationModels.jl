_unwrap(wrapper::SemFiniteDiff) = wrapper.model
params(wrapper::SemFiniteDiff) = params(wrapper.model)
loss_terms(wrapper::SemFiniteDiff) = loss_terms(wrapper.model)

replace_observed(wrapper::SemFiniteDiff, data; kwargs...) =
    SemFiniteDiff(replace_observed(wrapper.model, data; kwargs...))

FiniteDiffLossWrappers = Union{LossFiniteDiff, SemLossFiniteDiff}

_unwrap(term::AbstractLoss) = term
_unwrap(wrapper::FiniteDiffLossWrappers) = wrapper.loss
implied(wrapper::FiniteDiffLossWrappers) = implied(_unwrap(wrapper))
observed(wrapper::FiniteDiffLossWrappers) = observed(_unwrap(wrapper))

replace_observed(wrapper::LossFiniteDiff, data; kwargs...) =
    LossFiniteDiff(replace_observed(_unwrap(wrapper), data; kwargs...))

replace_observed(wrapper::SemLossFiniteDiff, new_observed::SemObserved; kwargs...) =
    SemLossFiniteDiff(replace_observed(_unwrap(wrapper), new_observed; kwargs...))

replace_observed(
    wrapper::SemLossFiniteDiff,
    data::Union{AbstractMatrix, DataFrame};
    kwargs...,
) = SemLossFiniteDiff(replace_observed(_unwrap(wrapper), data; kwargs...))

FiniteDiffWrapper(model::AbstractSem) = SemFiniteDiff(model)
FiniteDiffWrapper(loss::AbstractLoss) = LossFiniteDiff(loss)
FiniteDiffWrapper(loss::SemLoss) = SemLossFiniteDiff(loss)

function evaluate!(
    objective,
    gradient,
    hessian,
    sem::Union{SemFiniteDiff, FiniteDiffLossWrappers},
    params,
)
    wrapped = _unwrap(sem)
    obj(p) = _evaluate!(
        objective_zero(objective, gradient, hessian),
        nothing,
        nothing,
        wrapped,
        p,
    )
    isnothing(gradient) || FiniteDiff.finite_difference_gradient!(gradient, obj, params)
    isnothing(hessian) || FiniteDiff.finite_difference_hessian!(hessian, obj, params)
    # FIXME if objective is not calculated, SemLoss implied states may not correspond to params
    return !isnothing(objective) ? obj(params) : nothing
end
