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

# don't change non-SEM terms
replace_observed(loss::AbstractLoss; kwargs...) = loss

# use the same observed type as before
replace_observed(loss::SemLoss; kwargs...) =
    replace_observed(loss, typeof(SEM.observed(loss)).name.wrapper; kwargs...)

# construct a new observed type
replace_observed(loss::SemLoss, observed_type; kwargs...) =
    replace_observed(loss, observed_type(; kwargs...); kwargs...)

function replace_observed(loss::SemLoss, new_observed::SemObserved; kwargs...)
    kwargs = Dict{Symbol, Any}(kwargs...)
    old_observed = SEM.observed(loss)
    implied = SEM.implied(loss)

    # get field types
    kwargs[:observed_type] = typeof(new_observed)
    kwargs[:old_observed_type] = typeof(old_observed)

    # update implied
    new_implied = update_observed(implied, new_observed; kwargs...)
    kwargs[:implied] = new_implied
    kwargs[:implied_type] = typeof(new_implied)
    kwargs[:nparams] = nparams(new_implied)

    # update loss
    return update_observed(loss, new_observed; kwargs...)
end

replace_observed(loss::LossTerm; kwargs...) =
    LossTerm(replace_observed(loss.loss; kwargs...), loss.id, loss.weight)

function replace_observed(sem::Sem; kwargs...)
    updated_terms = Tuple(replace_observed(term; kwargs...) for term in loss_terms(sem))
    return Sem(updated_terms...)
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
            group =>
                select(filter(r -> r[column] == group, kwargs[:data]), Not(column)) for
            group in emodel.groups
        )
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
    rand(sem::Union{Sem, SemLoss, SemImplied}, params, n)
    rand(sem::Union{Sem, SemLoss, SemImplied}, n)

Sample from the multivariate normal distribution implied by the SEM model.

# Arguments
- `sem`: SEM model to use. Ensemble models with multiple SEM terms are not supported.
- `params`: SEM model parameters to simulate from.
- `n::Integer`: Number of samples to draw.

# Examples
```julia
rand(model, start_simple(model), 100)
```
"""
function Distributions.rand(implied::SemImplied, params, n::Integer)
    update!(EvaluationTargets{true, false, false}(), implied, params)
    return rand(implied, n)
end

Distributions.rand(implied::SemImplied, n::Integer) =
    error("rand($(typeof(implied)), n) is not implemented")

function Distributions.rand(implied::Union{RAM, RAMSymbolic}, n::Integer)
    Σ = Symmetric(implied.Σ)
    if MeanStruct(implied) === NoMeanStruct
        return permutedims(rand(MvNormal(Σ), n))
    elseif MeanStruct(implied) === HasMeanStruct
        return permutedims(rand(MvNormal(implied.μ, Σ), n))
    end
end

Distributions.rand(loss::SemLoss, params, n::Integer) = rand(SEM.implied(loss), params, n)

Distributions.rand(loss::SemLoss, n::Integer) = rand(SEM.implied(loss), n)

Distributions.rand(model::Sem, params, n::Integer) = rand(sem_term(model), params, n)

Distributions.rand(model::Sem, n::Integer) = rand(sem_term(model), n)
