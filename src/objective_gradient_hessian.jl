############################################################################################
# methods for AbstractSem
############################################################################################

function objective!(model::AbstractSemSingle, parameters)
    objective!(imply(model), parameters, model)
    return objective!(loss(model), parameters, model)
end

function gradient!(gradient, model::AbstractSemSingle, parameters)
    fill!(gradient, zero(eltype(gradient)))
    gradient!(imply(model), parameters, model)
    gradient!(gradient, loss(model), parameters, model)
end

function hessian!(hessian, model::AbstractSemSingle, parameters)
    fill!(hessian, zero(eltype(hessian)))
    hessian!(imply(model), parameters, model)
    hessian!(hessian, loss(model), parameters, model)
end

function objective_gradient!(gradient, model::AbstractSemSingle, parameters)
    fill!(gradient, zero(eltype(gradient)))
    objective_gradient!(imply(model), parameters, model)
    objective_gradient!(gradient, loss(model), parameters, model)
end

function objective_hessian!(hessian, model::AbstractSemSingle, parameters)
    fill!(hessian, zero(eltype(hessian)))
    objective_hessian!(imply(model), parameters, model)
    objective_hessian!(hessian, loss(model), parameters, model)
end

function gradient_hessian!(gradient, hessian, model::AbstractSemSingle, parameters)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    gradient_hessian!(imply(model), parameters, model)
    gradient_hessian!(gradient, hessian, loss(model), parameters, model)
end

function objective_gradient_hessian!(gradient, hessian, model::AbstractSemSingle, parameters)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    objective_gradient_hessian!(imply(model), parameters, model)
    return objective_gradient_hessian!(gradient, hessian, loss(model), parameters, model)
end

############################################################################################
# methods for SemFiniteDiff and SemForwardDiff
############################################################################################

# gradient methods call themselves with the additional model.has_gradient argument

gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par) = 
    gradient!(gradient, model, par, model.has_gradient)

objective_gradient!(gradient, model::Union{SemFiniteDiff, SemForwardDiff}, par) = 
    objective_gradient!(gradient, model, par, model.has_gradient)

# methods where autodiff takes place 
# - these are specific to the method of automatic differentiation

# FiniteDiff
gradient!(gradient, model::SemFiniteDiff, par, has_gradient::Val{false}) =
    FiniteDiff.finite_difference_gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemFiniteDiff, par) = 
    FiniteDiff.finite_difference_hessian!(hessian, x -> objective!(model, x), par)

# ForwardDiff
gradient!(gradient, model::SemForwardDiff, par, has_gradient::Val{false}) =
    ForwardDiff.gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemForwardDiff, par) = 
    ForwardDiff.hessian!(hessian, x -> objective!(model, x), par)

# gradient!
function gradient!(
        gradient, 
        model::Union{SemFiniteDiff, SemForwardDiff}, 
        parameters, 
        has_gradient::Val{true})
    fill!(gradient, zero(eltype(gradient)))
    gradient!(imply(model), parameters, model)
    gradient!(gradient, loss(model), parameters, model)
end

# objective_gradient!
function objective_gradient!(
        gradient, 
        model::Union{SemFiniteDiff, SemForwardDiff}, 
        parameters, 
        has_gradient::Val{true})
    fill!(gradient, zero(eltype(gradient)))
    objective_gradient!(imply(model), parameters, model)
    return objective_gradient!(gradient, loss(model), parameters, model)
end

function objective_gradient!(
        gradient, 
        model::Union{SemFiniteDiff, SemForwardDiff}, 
        parameters, 
        has_gradient::Val{false})
    fill!(gradient, zero(eltype(gradient)))
    gradient!(gradient, model, parameters)
    return objective!(model, parameters)
end

# other methods
function gradient_hessian!(
        gradient, 
        hessian, 
        model::Union{SemFiniteDiff, SemForwardDiff}, 
        parameters)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    gradient!(gradient, model, parameters)
    hessian!(hessian, model, parameters)
end

function objective_hessian!(hessian, model::Union{SemFiniteDiff, SemForwardDiff}, parameters)
    fill!(hessian, zero(eltype(hessian)))
    hessian!(hessian, model, parameters)
    return objective!(model, parameters)
end

function objective_gradient_hessian!(
        gradient, 
        hessian, 
        model::Union{SemFiniteDiff, SemForwardDiff}, 
        parameters)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    hessian!(hessian, model, parameters)
    return objective_gradient!(gradient, model, parameters)
end

############################################################################################
# methods for SemLoss
############################################################################################

function objective!(loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> fun_weight[2]*objective!(fun_weight[1], par, model), 
        +, 
        zip(loss.functions, loss.weights)
        )
end

function gradient!(gradient, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        new_gradient = gradient!(lossfun, par, model)
        gradient .+= w*new_gradient
    end
end

function hessian!(hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        hessian .+= w*hessian!(lossfun, par, model)
    end
end

function objective_gradient!(gradient, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_gradient_wrap_(gradient, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

function objective_hessian!(hessian, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_hessian_wrap_(hessian, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

function gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    for (lossfun, w) in zip(loss.functions, loss.weights)
        new_gradient, new_hessian = gradient_hessian!(lossfun, par, model)
        gradient .+= w*new_gradient
        hessian .+= w*new_hessian
    end
end

function objective_gradient_hessian!(gradient, hessian, loss::SemLoss, par, model)
    return mapreduce(
        fun_weight -> objective_gradient_hessian_wrap_(gradient, hessian, fun_weight[1], par, model, fun_weight[2]),
        +, 
        zip(loss.functions, loss.weights)
        )
end

# wrapper to update gradient/hessian and return objective value
function objective_gradient_wrap_(gradient, lossfun, par, model, w)
    new_objective, new_gradient = objective_gradient!(lossfun, par, model)
    gradient .+= w*new_gradient
    return w*new_objective
end

function objective_hessian_wrap_(hessian, lossfun, par, model, w)
    new_objective, new_hessian = objective_hessian!(lossfun, par, model)
    hessian .+= w*new_hessian
    return w*new_objective
end

function objective_gradient_hessian_wrap_(gradient, hessian, lossfun, par, model, w)
    new_objective, new_gradient, new_hessian = objective_gradient_hessian!(lossfun, par, model)
    gradient .+= w*new_gradient
    hessian .+= w*new_hessian
    return w*new_objective
end

############################################################################################
# methods for SemEnsemble
############################################################################################

function objective!(ensemble::SemEnsemble, par)
    return mapreduce(
        model_weight -> model_weight[2]*objective!(model_weight[1], par), 
        +, 
        zip(ensemble.sems, ensemble.weights)
        )
end

function gradient!(gradient, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)
        gradient_new = similar(gradient)
        gradient!(gradient_new, model, par)
        gradient .+= w*gradient_new
    end
end

function hessian!(hessian, ensemble::SemEnsemble, par)
    fill!(hessian, zero(eltype(hessian)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)
        hessian_new = similar(hessian)
        hessian!(hessian_new, model, par)
        hessian .+= w*hessian_new
    end
end

function objective_gradient!(gradient, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    return mapreduce(
        model_weight -> objective_gradient_wrap_(gradient, model_weight[1], par, model_weight[2]),
        +, 
        zip(ensemble.sems, ensemble.weights)
        )
end

function objective_hessian!(hessian, ensemble::SemEnsemble, par)
    fill!(hessian, zero(eltype(hessian)))
    return mapreduce(
        model_weight -> objective_hessian_wrap_(hessian, model_weight[1], par, model_weight[2]),
        +,
        zip(ensemble.sems, ensemble.weights)
        )
end

function gradient_hessian!(gradient, hessian, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    for (model, w) in zip(ensemble.sems, ensemble.weights)

        new_gradient = similar(gradient)
        new_hessian = similar(hessian)

        gradient_hessian!(new_gradient, new_hessian, model, par)

        gradient .+= w*new_gradient
        hessian .+= w*new_hessian

    end
end

function objective_gradient_hessian!(gradient, hessian, ensemble::SemEnsemble, par)
    fill!(gradient, zero(eltype(gradient)))
    fill!(hessian, zero(eltype(hessian)))
    return mapreduce(
        model_weight -> objective_gradient_hessian_wrap_(gradient, hessian, model_weight[1], par, model, model_weight[2]),
        +, 
        zip(ensemble.sems, ensemble.weights)
        )
end

# wrapper to update gradient/hessian and return objective value
function objective_gradient_wrap_(gradient, model::AbstractSemSingle, par, w)
    gradient_pre = similar(gradient)
    new_objective = objective_gradient!(gradient_pre, model, par)
    gradient .+= w*gradient_pre
    return w*new_objective
end

function objective_hessian_wrap_(hessian, model::AbstractSemSingle, par, w)
    hessian_pre = similar(hessian)
    new_objective = objective_hessian!(hessian_pre, model, par)
    hessian .+= w*new_hessian
    return w*new_objective
end

function objective_gradient_hessian_wrap_(gradient, hessian, model::AbstractSemSingle, par, w)
    gradient_pre = similar(gradient)
    hessian_pre = similar(hessian)
    new_objective = objective_gradient_hessian!(gradient_pre, hessian_pre, model, par)
    gradient .+= w*new_gradient
    hessian .+= w*new_hessian
    return w*new_objective
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
    objective!(model::AbstractSem, parameters)

Returns the objective value at `parameters`.
The model object can be modified.

# Implementation
To implement a new `SemImply` or `SemLossFunction` subtype, you need to add a method for
    objective!(newtype::MyNewType, parameters, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you need to add a method for
    objective!(model::MyNewType, parameters)
"""
function objective! end

"""
    gradient!(gradient, model::AbstractSem, parameters)

Writes the gradient value at `parameters` to `gradient`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    gradient!(newtype::MyNewType, parameters, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    gradient!(gradient, model::MyNewType, parameters)
"""
function gradient! end

"""
    hessian!(hessian, model::AbstractSem, parameters)

Writes the hessian value at `parameters` to `hessian`.

# Implementation
To implement a new `SemImply` or `SemLossFunction` type, you can add a method for
    hessian!(newtype::MyNewType, parameters, model::AbstractSemSingle)

To implement a new `AbstractSem` subtype, you can add a method for
    hessian!(hessian, model::MyNewType, parameters)
"""
function hessian! end