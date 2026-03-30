_unwrap(wrapper::SemFiniteDiff) = wrapper.model
params(wrapper::SemFiniteDiff) = params(wrapper.model)
loss_terms(wrapper::SemFiniteDiff) = loss_terms(wrapper.model)

replace_observed(wrapper::SemFiniteDiff, data) =
    SemFiniteDiff(replace_observed(wrapper.model, data))

FiniteDiffLossWrappers = Union{LossFiniteDiff, SemLossFiniteDiff}

_unwrap(term::AbstractLoss) = term
_unwrap(wrapper::FiniteDiffLossWrappers) = wrapper.loss
implied(wrapper::FiniteDiffLossWrappers) = implied(_unwrap(wrapper))
observed(wrapper::FiniteDiffLossWrappers) = observed(_unwrap(wrapper))

replace_observed(wrapper::LossFiniteDiff, data) =
    LossFiniteDiff(replace_observed(_unwrap(wrapper), data))

replace_observed(wrapper::SemLossFiniteDiff, new_observed::SemObserved) =
    SemLossFiniteDiff(replace_observed(_unwrap(wrapper), new_observed))

replace_observed(
    wrapper::SemLossFiniteDiff,
    data::Union{AbstractMatrix, DataFrame},
) = SemLossFiniteDiff(replace_observed(_unwrap(wrapper), data))

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
