############################################################################################
# Define the basic type system
############################################################################################
"Most abstract supertype for all SEMs"
abstract type AbstractSem end

"Supertype for all single SEMs, e.g. SEMs that have at least the fields `observed`, `imply`, loss and diff"
abstract type AbstractSemSingle{O, I, L, D} <: AbstractSem end

"Supertype for all collections of multiple SEMs"
abstract type AbstractSemCollection <: AbstractSem end

"Supertype for all loss functions of SEMs. If you want to implement a custom loss function, it should be a subtype of `SemLossFunction`."
abstract type SemLossFunction end

"""
    SemLoss(args...; loss_weights = nothing, ...)

Constructs the loss field of a SEM. Can contain multiple `SemLossFunction`s, the model is optimized over their sum.
See also [`SemLossFunction`](@ref).

# Arguments
- `args...`: Multiple `SemLossFunction`s.
- `loss_weights::Vector`: Weights for each loss function. Defaults to unweighted optimization.

# Examples
```julia
my_ml_loss = SemML(...)
my_ridge_loss = SemRidge(...)
my_loss = SemLoss(SemML, SemRidge; loss_weights = [1.0, 2.0])
```
"""
mutable struct SemLoss{F <: Tuple, T}
    functions::F
    weights::T
end

function SemLoss(functions...; loss_weights = nothing, kwargs...)

    if !isnothing(loss_weights) 
        loss_weights = SemWeight.(loss_weights)
    else
        loss_weights = Tuple(SemWeight(nothing) for _ in 1:length(functions))
    end

    return SemLoss(
        functions,
        loss_weights
        )
end

# weights for loss functions or models. If the weight is nothing, multiplication returs second argument
struct SemWeight{T}
    w::T
end

Base.:*(x::SemWeight{Nothing}, y) = y
Base.:*(x::SemWeight, y) = x.w*y

"""
Supertype of all objects that can serve as the diff field of a SEM.
Connects the SEM to its optimization backend and controls options like the optimization algorithm.
If you want to connect the SEM package to a new optimization backend, you should implement a subtype of SemDiff.
"""
abstract type SemDiff end

"""
Supertype of all objects that can serve as the observed field of a SEM.
Pre-processes data and computes sufficient statistics for example.
If you have a special kind of data, e.g. ordinal data, you should implement a subtype of SemObs.
"""
abstract type SemObs end

"""
Supertype of all objects that can serve as the imply field of a SEM.
Computed model-implied values that should be compared with the observed data to find parameter estimates,
e. g. the model implied covariance or mean.
If you would like to implement a different notation, e.g. LISREL, you should implement a subtype of SemImply.
"""
abstract type SemImply end

"Subtype of SemImply for all objects that can serve as the imply field of a SEM and use some form of symbolic precomputation."
abstract type SemImplySymbolic <: SemImply end

"""
    Sem(;observed = SemObsCommon, imply = RAM, loss = SemML, diff = SemDiffOptim, kwargs...)

Constructor for the basic `Sem` type.
All additional kwargs are passed down to the constructors for the observed, imply, loss and diff fields.

# Arguments
- `observed`: object of subtype `SemObs` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `diff`: object of subtype `SemDiff` or a constructor.

Returns a Sem with fields
- `observed::SemObs`: Stores observed data, sample statistics, etc. See also [`SemObs`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `diff::SemDiff`: Connects the model to the optimizer. See also [`SemDiff`](@ref).
"""
mutable struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff} <: AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I
    loss::L
    diff::D
end

############################################################################################
# automatic differentiation
############################################################################################
"""
    SemFiniteDiff(;observed = SemObsCommon, imply = RAM, loss = SemML, diff = SemDiffOptim, has_gradient = false, kwargs...)

Constructor for `SemFiniteDiff`.
All additional kwargs are passed down to the constructors for the observed, imply, loss and diff fields.

# Arguments
- `observed`: object of subtype `SemObs` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `diff`: object of subtype `SemDiff` or a constructor.
- `has_gradient::Bool`: are analytic gradients available for this model.

Returns a Sem with fields
- `observed::SemObs`: Stores observed data, sample statistics, etc. See also [`SemObs`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `diff::SemDiff`: Connects the model to the optimizer. See also [`SemDiff`](@ref).
- `has_gradient::Val{Bool}`: signifies if analytic gradients are available for this model.
"""
struct SemFiniteDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I
    loss::L
    diff::D
    has_gradient::G
end

"""
    SemForwardDiff(;observed = SemObsCommon, imply = RAM, loss = SemML, diff = SemDiffOptim, has_gradient = false, kwargs...)

Constructor for `SemForwardDiff`.
All additional kwargs are passed down to the constructors for the observed, imply, loss and diff fields.

# Arguments
- `observed`: object of subtype `SemObs` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `diff`: object of subtype `SemDiff` or a constructor.
- `has_gradient::Bool`: are analytic gradients available for this model.

Returns a Sem with fields
- `observed::SemObs`: Stores observed data, sample statistics, etc. See also [`SemObs`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `diff::SemDiff`: Connects the model to the optimizer. See also [`SemDiff`](@ref).
- `has_gradient::Val{Bool}`: signifies if analytic gradients are available for this model.
"""
struct SemForwardDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I 
    loss::L 
    diff::D
    has_gradient::G
end

############################################################################################
# ensemble models
############################################################################################
"""
    SemEnsemble(models..., diff = SemDiffOptim, weights = nothing, kwargs...)

Constructor for `SemForwardDiff`.
All additional kwargs are passed down to the constructor for the diff field.

# Arguments
- `models...`: `AbstractSem`s.
- `diff`: object of subtype `SemDiff` or a constructor.
- `weights::Vector`:  Weights for each model. Defaults to the number of observed data points.

Returns a SemEnsemble with fields
- `n::Int`: Number of models.
- `sems::Tuple`: `AbstractSem`s.
- `weights::Vector`: Weights for each model.
- `diff::SemDiff`: Connects the model to the optimizer. See also [`SemDiff`](@ref).
- `identifier::Dict`: Stores parameter labels and their position.
"""
struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, D, I} <: AbstractSemCollection
    n::N
    sems::T
    weights::V
    diff::D
    identifier::I
end

function SemEnsemble(models...; diff = SemDiffOptim, weights = nothing, kwargs...)
    n = length(models)
    npar = n_par(models[1])

    # default weights
    
    if isnothing(weights)
        nobs_total = sum(n_obs.(models))
        weights = [n_obs(model)/nobs_total for model in models]
    end

    # check identifier equality
    id = identifier(models[1])
    for model in models
        if id != identifier(model)
            throw(ErrorException("The identifier of your models do not match. \n
            Maybe you tried to specify models of an ensemble via ParameterTables. \n
            In that case, you may use RAMMatrices instead."))
        end
    end

    # diff
    if !isa(diff, SemDiff)
        diff = diff(;kwargs...)
    end

    return SemEnsemble(
        n,
        models,
        weights,
        diff,
        id
        )
end

"""
    n_models(ensemble::SemEnsemble) -> Integer

Returns the number of models in an ensemble model.
"""
n_models(ensemble::SemEnsemble) = ensemble.n
"""
    models(ensemble::SemEnsemble) -> Tuple{AbstractSem}

Returns the models in an ensemble model.
"""
models(ensemble::SemEnsemble) = ensemble.sems
"""
    weights(ensemble::SemEnsemble) -> Vector

Returns the weights of an ensemble model.
"""
weights(ensemble::SemEnsemble) = ensemble.weights
"""
    diff(ensemble::SemEnsemble) -> SemDiff

Returns the diff part of an ensemble model.
"""
diff(ensemble::SemEnsemble) = ensemble.diff

############################################################################################
# additional methods
############################################################################################
"""
    observed(model::AbstractSemSingle) -> SemObs

Returns the observed part of a model.
"""
observed(model::AbstractSemSingle) = model.observed

"""
    imply(model::AbstractSemSingle) -> SemImply

Returns the imply part of a model.
"""
imply(model::AbstractSemSingle) = model.imply

"""
    loss(model::AbstractSemSingle) -> SemLoss

Returns the loss part of a model.
"""
loss(model::AbstractSemSingle) = model.loss

"""
    diff(model::AbstractSemSingle) -> SemDiff

Returns the diff part of a model.
"""
diff(model::AbstractSemSingle) = model.diff

"""
    diff(model::AbstractSemSingle) -> Val{bool}

Returns whether the model has analytic gradients.
"""
has_gradient(model::SemForwardDiff) = model.has_gradient
has_gradient(model::SemFiniteDiff) = model.has_gradient