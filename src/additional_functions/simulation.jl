"""
    (1) replace_observed(model::AbstractSemSingle; kwargs...)

    (2) replace_observed(model::AbstractSemSingle, observed; kwargs...)

    (3) replace_observed(model::SemEnsemble; column = :group, weights = nothing, kwargs...)

Return a new model with swaped observed part.

# Arguments
- `model::AbstractSemSingle`: model to swap the observed part of.
- `kwargs`: additional keyword arguments; typically includes `data` and `specification`
- `observed`: Either an object of subtype of `SemObserved` or a subtype of `SemObserved`

# For SemEnsemble models:
- `column`: if a DataFrame is passed as `data = ...`, which column signifies the group?
- `weights`: how to weight the different sub-models,
    defaults to number of samples per group in the new data
- `kwargs`: has to be a dict with keys equal to the group names.
    For `data` can also be a DataFrame with `column` containing the group information,
    and for `specification` can also be an `EnsembleParameterTable`.

# Examples
See the online documentation on [Replace observed data](@ref).
"""
function replace_observed end

"""
    update_observed(to_update, observed::SemObserved; kwargs...)

Update a `SemImplied`, `SemLossFunction` or `SemOptimizer` object to use a `SemObserved` object.

# Examples
See the online documentation on [Replace observed data](@ref).

# Implementation
You can provide a method for this function when defining a new type, for more information
on this see the online developer documentation on [Update observed data](@ref).
"""
function update_observed end

############################################################################################
# change observed (data) without reconstructing the whole model
############################################################################################

# use the same observed type as before
replace_observed(model::AbstractSemSingle; kwargs...) =
    replace_observed(model, typeof(observed(model)).name.wrapper; kwargs...)

function replace_observed(model::AbstractSemSingle, observed_type; kwargs...)
    new_observed = observed_type(; kwargs...)
    kwargs = Dict{Symbol, Any}(kwargs...)

    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(model.observed)
    kwargs[:implied_type] = typeof(model.implied)
    kwargs[:loss_types] = [typeof(lossfun) for lossfun in model.loss.functions]

    # update implied
    new_implied = update_observed(model.implied, new_observed; kwargs...)
    kwargs[:implied] = new_implied
    kwargs[:nparams] = nparams(new_implied)

    # update loss
    new_loss = update_observed(model.loss, new_observed; kwargs...)

    return Sem(
        new_observed,
        new_implied,
        new_loss,
    )
end

function update_observed(loss::SemLoss, new_observed; kwargs...)
    new_functions = Tuple(
        update_observed(lossfun, new_observed; kwargs...) for lossfun in loss.functions
    )
    return SemLoss(new_functions, loss.weights)
end

function replace_observed(
    emodel::SemEnsemble;
    column = :group,
    weights = nothing,
    kwargs...,
)
    kwargs = Dict{Symbol, Any}(kwargs...)
    # allow for EnsembleParameterTable to be passed as specification
    if haskey(kwargs, :specification) && isa(kwargs[:specification], EnsembleParameterTable)
        kwargs[:specification] = convert(Dict{Symbol, RAMMatrices}, kwargs[:specification])
    end
    # allow for DataFrame with group variable "column" to be passed as new data
    if haskey(kwargs, :data) && isa(kwargs[:data], DataFrame)
        kwargs[:data] = Dict(
            group => select(
                filter(
                    r -> r[column] == group,
                    kwargs[:data]),
                Not(column)) for group in emodel.groups)
    end
    # update each model for new data
    models = emodel.sems
    new_models = Tuple(
        replace_observed(m; group_kwargs(g, kwargs)...) for
        (m, g) in zip(models, emodel.groups)
    )
    return SemEnsemble(new_models...; weights = weights, groups = emodel.groups)
end

function group_kwargs(g, kwargs)
    return Dict(k => kwargs[k][g] for k in keys(kwargs))
end

############################################################################################
# simulate data
############################################################################################
"""
    (1) rand(model::AbstractSemSingle, params, n)

    (2) rand(model::AbstractSemSingle, n)

Sample normally distributed data from the model-implied covariance matrix and mean vector.

# Arguments
- `model::AbstractSemSingle`: model to simulate from.
- `params`: parameter values to simulate from.
- `n::Integer`: Number of samples.

# Examples
```julia
rand(model, start_simple(model), 100)
```
"""
function Distributions.rand(
    model::AbstractSemSingle{O, I, L},
    params,
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L}
    update!(EvaluationTargets{true, false, false}(), model.implied, model, params)
    return rand(model, n)
end

function Distributions.rand(
    model::AbstractSemSingle{O, I, L},
    n::Integer,
) where {O, I <: Union{RAM, RAMSymbolic}, L}
    if MeanStruct(model.implied) === NoMeanStruct
        data = permutedims(rand(MvNormal(Symmetric(model.implied.Σ)), n))
    elseif MeanStruct(model.implied) === HasMeanStruct
        data = permutedims(rand(MvNormal(model.implied.μ, Symmetric(model.implied.Σ)), n))
    end
    return data
end
