############################################################################################
# Define the basic type system
############################################################################################

"Meanstructure trait for `SemImplied` subtypes"
abstract type MeanStruct end
"Indicates that `SemImplied` subtype supports mean structure"
struct HasMeanStruct <: MeanStruct end
"Indicates that `SemImplied` subtype does not support mean structure"
struct NoMeanStruct <: MeanStruct end

# default implementation
MeanStruct(::Type{T}) where {T} =
    hasfield(T, :meanstruct) ? fieldtype(T, :meanstruct) :
    error("Objects of type $T do not support MeanStruct trait")

MeanStruct(semobj) = MeanStruct(typeof(semobj))

"Hessian Evaluation trait for `SemImplied` and `SemLossFunction` subtypes"
abstract type HessianEval end
struct ApproxHessian <: HessianEval end
struct ExactHessian <: HessianEval end

# default implementation
HessianEval(::Type{T}) where {T} =
    hasfield(T, :hessianeval) ? fieldtype(T, :hessianeval) :
    error("Objects of type $T do not support HessianEval trait")

HessianEval(semobj) = HessianEval(typeof(semobj))

"Supertype for all loss functions of SEMs. If you want to implement a custom loss function, it should be a subtype of `AbstractLoss`."
abstract type AbstractLoss end

abstract type SemOptimizer{E} end

# wrapper around optimization result
abstract type SemOptimizerResult{O <: SemOptimizer} end

"""
    abstract type SemObserved

Supertype of all objects that can serve as the observed field of a SEM.
Pre-processes data and computes sufficient statistics for example.
If you have a special kind of data, e.g. ordinal data, you should implement a subtype of SemObserved.
"""
abstract type SemObserved end

"""
Supertype of all objects that can serve as the implied field of a SEM.
Computes model-implied values that should be compared with the observed data to find parameter estimates,
e. g. the model implied covariance or mean.
If you would like to implement a different notation, e.g. LISREL, you should implement a subtype of SemImplied.
"""
abstract type SemImplied end

"Subtype of SemImplied for all objects that can serve as the implied field of a SEM and use some form of symbolic precomputation."
abstract type SemImpliedSymbolic <: SemImplied end

"""
    abstract type SemLoss{O <: SemObserved, I <: SemImplied} <: AbstractLoss

The base type for calculating the loss of the implied SEM model when explaining the observed data.

All subtypes of `SemLoss` should have the following fields:
- `observed::O`: object of subtype [`SemObserved`](@ref).
- `implied::I`: object of subtype [`SemImplied`](@ref).
"""
abstract type SemLoss{O <: SemObserved, I <: SemImplied} <: AbstractLoss end

"""
    abstract type AbstractSem

The base type for all SEMs.
"""
abstract type AbstractSem end

"""
    struct LossTerm{L, I, W}

A term of a [`Sem`](@ref) model that wraps [`AbstractLoss`](@ref) loss function of type `L`.
Loss term can have an optional *id* of type `I` and *weight* of numeric type `W`.
"""
struct LossTerm{L <: AbstractLoss, I <: Union{Symbol, Nothing}, W <: Union{Number, Nothing}}
    loss::L
    id::I
    weight::W
end

"""
    Sem(loss_terms...; [params], kwargs...)

SEM model (including multi-group SEMs) that combines all the data, implied SEM structure
and regularization terms.

All terms of the `Sem` object share the same set of parameters.
`Sem` implements the calculation of the weighted sum of its terms (the *objective*
function), as well as the gradient and Hessian of this sum.

# Arguments
- `loss_terms...`: [`AbstractLoss`](@ref) objects, including SEM losses ([`SemLoss`](@ref)),
  optionally can be a pair of a loss object and its numeric weight

# Fields
- `loss_terms::Tuple`: a tuple of all loss functions and their weights
- `params::Vector{Symbol}`: the vector of parameter ids shared by all loss functions.
"""
struct Sem{L <: Tuple} <: AbstractSem
    loss_terms::L
    params::Vector{Symbol}
end

############################################################################################
# automatic differentiation
############################################################################################

"""
    SemFiniteDiff(model::AbstractSem)

A wrapper around [`AbstractSem`](@ref) that substitutes dedicated evaluation of gradient and
hessian with finite difference approximation.

`SemFiniteDiff` could be used to enable gradient-based optimization of the SEM models
when the dedicated calculation of gradient and hessian are not available.
For approximation, it uses the *FiniteDiff.jl* package.

# Arguments
- `model::Sem`: the SEM model to wrap
"""
struct SemFiniteDiff{S <: AbstractSem} <: AbstractSem
    model::S
end

struct LossFiniteDiff{L <: AbstractLoss} <: AbstractLoss
    loss::L
end

struct SemLossFiniteDiff{O, I, L <: SemLoss{O, I}} <: SemLoss{O, I}
    loss::L
end

"""
    abstract type SemSpecification end

Base type for all SEM specifications.
"""
abstract type SemSpecification end

abstract type AbstractParameterTable <: SemSpecification end
