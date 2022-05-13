#####################################################################################################
# Define the basic type system
#####################################################################################################
"Most abstract supertype for all SEMs"
abstract type AbstractSem end

"Supertype for all single SEMs, e.g. SEMs that have the fields `observed`, `imply`, loss, diff"
abstract type AbstractSemSingle <: AbstractSem end

"Supertype for all collections of multiple SEMs"
abstract type AbstractSemCollection <: AbstractSem end

"Supertype for all loss functions of SEMs. If you want to implement a custom loss function, it should be of this type."
abstract type SemLossFunction end

"""
    SemLoss(args...; ...)

Constructs the loss field of a SEM. Can contain multiple `SemLossFunction`s, the model is optimized over their sum.
See also [`SemLossFunction`](@ref).

# Arguments
- `args...`: Multiple `SemLossFunction`s.

# Examples
```julia
my_ml_loss = SemML(...)
my_ridge_loss = SemRidge(...)
my_loss = SemLoss(SemML, SemRidge)
```
"""
mutable struct SemLoss{F <: Tuple, T, FT, GT, HT}
    functions::F
    weights::T
end

function SemLoss(functions...; loss_weights = nothing, kwargs...)

    if !isnothing(loss_weights) 
        loss_weights = SemWeight.(loss_weights)
    else
        loss_weights = Tuple(SemWeight(nothing for _ in 1:length(functions)))

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
    Sem(;observed = SemObsCommon, imply = RAM, loss = (SemML,), diff = SemDiffOptim, kwargs...)

Constructor for the basic `Sem` type.
All additional kwargs are passed down to the constructors for the fields.

# Arguments
- `observed`: isa `SemObs` or a constructor.
- `imply`: isa `SemImply` or a constructor.
- `loss`: Tuple of objects that are `SemLossFunction`s or constructors.
- `observed`: isa `SemObs` or a constructor.

Returns a struct of type Sem with fields
- `observed::SemObs`: Stores observed data, sample statistics, etc. See also [`SemObs`](@ref).
- `imply::SemImply`: Computes Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `diff::SemDiff`: Connects the model to the optimizer. See also [`SemDiff`](@ref).
"""
mutable struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff} <: AbstractSemSingle
    observed::O
    imply::I
    loss::L
    diff::D
end

#####################################################################################################
# automatic differentiation
#####################################################################################################

struct SemFiniteDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSemSingle
    observed::O
    imply::I
    loss::L
    diff::D
    has_gradient::G
end

struct SemForwardDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSemSingle
    observed::O
    imply::I 
    loss::L 
    diff::D
    has_gradient::G
end

#####################################################################################################
# ensemble models
#####################################################################################################

struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, D, I, FT, GT, HT} <: AbstractSemCollection
    n::N
    sems::T
    weights::V
    diff::D
    identifier::I
end

function SemEnsemble(models...; diff = SemDiffOptim, weights = nothing, parameter_type = Float64, kwargs...)
    n = length(models)

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

function (ensemble::SemEnsemble)(par, F, G, H)

    if H ensemble.hessian .= 0.0 end
    if G ensemble.gradient .= 0.0 end
    if F ensemble.objective .= 0.0 end
            
    for (model, weight) in zip(ensemble.sems, ensemble.weights)
        model(par, F, G, H)
        if H ensemble.hessian .+= weight*hessian(model) end
        if G ensemble.gradient .+= weight*gradient(model) end
        if F ensemble.objective .+= weight*objective(model) end
    end

end

n_models(ensemble::SemEnsemble) = ensemble.n
models(ensemble::SemEnsemble) = ensemble.sems
weights(ensemble::SemEnsemble) = ensemble.weights
diff(ensemble::SemEnsemble) = ensemble.diff

#####################################################################################################
# additional methods
#####################################################################################################

observed(model::Sem) = model.observed
imply(model::Sem) = model.imply
loss(model::Sem) = model.loss
diff(model::Sem) = model.diff

observed(model::SemForwardDiff) = model.observed
imply(model::SemForwardDiff) = model.imply
loss(model::SemForwardDiff) = model.loss
diff(model::SemForwardDiff) = model.diff
has_gradient(model::SemForwardDiff) = model.has_gradient

observed(model::SemFiniteDiff) = model.observed
imply(model::SemFiniteDiff) = model.imply
loss(model::SemFiniteDiff) = model.loss
diff(model::SemFiniteDiff) = model.diff
has_gradient(model::SemFiniteDiff) = model.has_gradient