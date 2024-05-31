############################################################################################
# methods for AbstractSem
############################################################################################

function objective!(model::AbstractSemSingle, params)
    objective!(imply(model), params, model)
    return objective!(loss(model), params, model)
end

function gradient!(gradient, model::AbstractSemSingle, params)
    fill!(gradient, zero(eltype(gradient)))
    gradient!(imply(model), params, model)
    gradient!(gradient, loss(model), params, model)
end

function hessian!(hessian, model::AbstractSemSingle, params)
    fill!(hessian, zero(eltype(hessian)))
    hessian!(imply(model), params, model)
    hessian!(hessian, loss(model), params, model)
end

function objective_gradient!(gradient, model::AbstractSemSingle, params)
    fill!(gradient, zero(eltype(gradient)))
    objective_gradient!(imply(model), params, model)
    objective_gradient!(gradient, loss(model), params, model)
end

function objective_hessian!(hessian, model::AbstractSemSingle, params)
    fill!(hessian, zero(eltype(hessian)))
    objective_hessian!(imply(model), params, model)
    objective_hessian!(hessian, loss(model), params, model)
end

function gradient_hessian!(gradient, hessian, model::AbstractSemSingle, params)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    gradient_hessian!(imply(model), params, model)
    gradient_hessian!(gradient, hessian, loss(model), params, model)
end

function objective_gradient_hessian!(gradient, hessian, model::AbstractSemSingle, params)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    objective_gradient_hessian!(imply(model), params, model)
    return objective_gradient_hessian!(gradient, hessian, loss(model), params, model)
end

############################################################################################
# methods for SemFiniteDiff
############################################################################################

gradient!(gradient, model::SemFiniteDiff, par) =
    FiniteDiff.finite_difference_gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemFiniteDiff, par) =
    FiniteDiff.finite_difference_hessian!(hessian, x -> objective!(model, x), par)

function objective_gradient!(gradient, model::SemFiniteDiff, params)
    gradient!(gradient, model, params)
    return objective!(model, params)
end

# other methods
function gradient_hessian!(gradient, hessian, model::SemFiniteDiff, params)
    gradient!(gradient, model, params)
    hessian!(hessian, model, params)
end

function objective_hessian!(hessian, model::SemFiniteDiff, params)
    hessian!(hessian, model, params)
    return objective!(model, params)
end

function objective_gradient_hessian!(gradient, hessian, model::SemFiniteDiff, params)
    hessian!(hessian, model, params)
    return objective_gradient!(gradient, model, params)
end

############################################################################################
# methods for SemLoss
############################################################################################

function objective!(loss::SemLoss, par, model)
    return mapreduce(
        (fun, weight) -> weight * objective!(fun, par, model),
        +,
        loss.functions,
        loss.weights,
    )
end

function gradient!(gradient, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        new_gradient = gradient!(lossfun, par, model)
        gradient .+= w * new_gradient
    end
end

function hessian!(hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        hessian .+= w * hessian!(lossfun, par, model)
    end
end

function objective_gradient!(gradient, loss::SemLoss, par, model)
    return mapreduce(
        (fun, weight) -> objective_gradient_wrap_(gradient, fun, par, model, weight),
        +,
        loss.functions,
        loss.weights,
    )
end

function objective_hessian!(hessian, loss::SemLoss, par, model)
    return mapreduce(
        (fun, weight) -> objective_hessian_wrap_(hessian, fun, par, model, weight),
        +,
        loss.functions,
        loss.weights,
    )
end

function gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        new_gradient, new_hessian = gradient_hessian!(lossfun, par, model)
        gradient .+= w * new_gradient
        hessian .+= w * new_hessian
    end
end

function objective_gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    return mapreduce(
        (fun, weight) ->
            objective_gradient_hessian_wrap_(gradient, hessian, fun, par, model, weight),
        +,
        loss.functions,
        loss.weights,
    )
end

# wrapper to update gradient/hessian and return objective value
function objective_gradient_wrap_(gradient, lossfun, par, model, w)
    new_objective, new_gradient = objective_gradient!(lossfun, par, model)
    gradient .+= w * new_gradient
    return w * new_objective
end

function objective_hessian_wrap_(hessian, lossfun, par, model, w)
    new_objective, new_hessian = objective_hessian!(lossfun, par, model)
    hessian .+= w * new_hessian
    return w * new_objective
end

function objective_gradient_hessian_wrap_(gradient, hessian, lossfun, par, model, w)
    new_objective, new_gradient, new_hessian =
        objective_gradient_hessian!(lossfun, par, model)
    gradient .+= w * new_gradient
    hessian .+= w * new_hessian
    return w * new_objective
end

############################################################################################
# methods for SemEnsemble
############################################################################################

function objective!(ensemble::SemEnsemble, par)
    return mapreduce(
        (model, weight) -> weight * objective!(model, par),
        +,
        ensemble.sems,
        ensemble.weights,
    )
end

function gradient!(gradient, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)
        gradient_new = similar(gradient)
        gradient!(gradient_new, model, par)
        gradient .+= w * gradient_new
    end
end

function hessian!(hessian, ensemble::SemEnsemble, par)
    fill!(hessian, zero(eltype(hessian)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)
        hessian_new = similar(hessian)
        hessian!(hessian_new, model, par)
        hessian .+= w * hessian_new
    end
end

function objective_gradient!(gradient, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    return mapreduce(
        (model, weight) -> objective_gradient_wrap_(gradient, model, par, weight),
        +,
        ensemble.sems,
        ensemble.weights,
    )
end

function objective_hessian!(hessian, ensemble::SemEnsemble, par)
    fill!(hessian, zero(eltype(hessian)))
    return mapreduce(
        (model, weight) -> objective_hessian_wrap_(hessian, model, par, weight),
        +,
        ensemble.sems,
        ensemble.weights,
    )
end

function gradient_hessian!(gradient, hessian, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)
        new_gradient = similar(gradient)
        new_hessian = similar(hessian)

        gradient_hessian!(new_gradient, new_hessian, model, par)

        gradient .+= w * new_gradient
        hessian .+= w * new_hessian
    end
end

function objective_gradient_hessian!(gradient, hessian, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    return mapreduce(
        (model, weight) ->
            objective_gradient_hessian_wrap_(gradient, hessian, model, par, model, weight),
        +,
        ensemble.sems,
        ensemble.weights,
    )
end

# wrapper to update gradient/hessian and return objective value
function objective_gradient_wrap_(gradient, model::AbstractSemSingle, par, w)
    gradient_pre = similar(gradient)
    new_objective = objective_gradient!(gradient_pre, model, par)
    gradient .+= w * gradient_pre
    return w * new_objective
end

function objective_hessian_wrap_(hessian, model::AbstractSemSingle, par, w)
    hessian_pre = similar(hessian)
    new_objective = objective_hessian!(hessian_pre, model, par)
    hessian .+= w * new_hessian
    return w * new_objective
end

function objective_gradient_hessian_wrap_(
    gradient,
    hessian,
    model::AbstractSemSingle,
    par,
    w,
)
    gradient_pre = similar(gradient)
    hessian_pre = similar(hessian)
    new_objective = objective_gradient_hessian!(gradient_pre, hessian_pre, model, par)
    gradient .+= w * new_gradient
    hessian .+= w * new_hessian
    return w * new_objective
end

############################################################################################
# generic methods for loss functions
############################################################################################

function objective_gradient!(lossfun::SemLossFunction, par, model)
    objective = objective!(lossfun::SemLossFunction, par, model)
    gradient = gradient!(lossfun::SemLossFunction, par, model)
    return objective, gradient
end

function objective_hessian!(lossfun::SemLossFunction, par, model)
    objective = objective!(lossfun::SemLossFunction, par, model)
    hessian = hessian!(lossfun::SemLossFunction, par, model)
    return objective, hessian
end

function gradient_hessian!(lossfun::SemLossFunction, par, model)
    gradient = gradient!(lossfun::SemLossFunction, par, model)
    hessian = hessian!(lossfun::SemLossFunction, par, model)
    return gradient, hessian
end

function objective_gradient_hessian!(lossfun::SemLossFunction, par, model)
    objective = objective!(lossfun::SemLossFunction, par, model)
    gradient = gradient!(lossfun::SemLossFunction, par, model)
    hessian = hessian!(lossfun::SemLossFunction, par, model)
    return objective, gradient, hessian
end

# throw an error by default if gradient! and hessian! are not implemented

#= gradient!(lossfun::SemLossFunction, par, model) =
    throw(ArgumentError("gradient for $(typeof(lossfun).name.wrapper) is not available"))

hessian!(lossfun::SemLossFunction, par, model) =
    throw(ArgumentError("hessian for $(typeof(lossfun).name.wrapper) is not available")) =#

############################################################################################
# generic methods for imply
############################################################################################

function objective_gradient!(semimp::SemImply, par, model)
    objective!(semimp::SemImply, par, model)
    gradient!(semimp::SemImply, par, model)
    return nothing
end

function objective_hessian!(semimp::SemImply, par, model)
    objective!(semimp::SemImply, par, model)
    hessian!(semimp::SemImply, par, model)
    return nothing
end

function gradient_hessian!(semimp::SemImply, par, model)
    gradient!(semimp::SemImply, par, model)
    hessian!(semimp::SemImply, par, model)
    return nothing
end

function objective_gradient_hessian!(semimp::SemImply, par, model)
    objective!(semimp::SemImply, par, model)
    gradient!(semimp::SemImply, par, model)
    hessian!(semimp::SemImply, par, model)
    return nothing
end

############################################################################################
# Documentation
############################################################################################
"""
    objective!(model::AbstractSem, params)

Returns the objective value at `params`.
The model object can be modified.

# Implementation
To implement a new `SemImply` or `SemLossFunction` subtype, you need to add a method for
    objective!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you need to add a method for
    objective!(model::MyNewType, params)
"""
function objective! end

"""
    gradient!(gradient, model::AbstractSem, params)

Writes the gradient value at `params` to `gradient`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    gradient!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    gradient!(gradient, model::MyNewType, params)
"""
function gradient! end

"""
    hessian!(hessian, model::AbstractSem, params)

Writes the hessian value at `params` to `hessian`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    hessian!(newtype::MyNewType, params, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    hessian!(hessian, model::MyNewType, params)
"""
function hessian! end
