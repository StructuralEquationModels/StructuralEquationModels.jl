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
mutable struct SemLoss{F <: Tuple, FT, GT, HT}
    functions::F

    F::FT
    G::GT
    H::HT
end

function SemLoss(functions; parameter_type = Float64)

    n_par = length(functions[1].G)

    return SemLoss(
        functions,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

SemLoss(args...; parameter_type = Float64) = SemLoss(args; parameter_type = parameter_type)

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

function (model::Sem)(par, F, G, H)
    model.imply(par, F, G, H, model)
    model.loss(par, F, G, H, model)
end

function (loss::SemLoss)(par, F, G, H, model)
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
    if H
        loss.H .= 0.0
        for lossfun in loss.functions
            loss.H .+= lossfun.H
        end
    end
    if G
        loss.G .= 0.0
        for lossfun in loss.functions
            loss.G .+= lossfun.G
        end
    end
    if F
        loss.F[1] = 0.0
        for lossfun in loss.functions
            loss.F[1] += lossfun.F[1]
        end
    end
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

function (model::SemFiniteDiff)(par, F, G, H)

    if H
        model.loss.H .= FiniteDiff.finite_difference_hessian(x -> objective!(model, x), par)
    end

    if model.has_gradient
        model.imply(par, F, G, false, model)
        model.loss(par, F, G, false, model)
    else

        if G
            model.loss.G .= FiniteDiff.finite_difference_gradient(x -> objective!(model, x), par)
        end

        if F
            model.imply(par, F, false, false, model)
            model.loss(par, F, false, false, model)
        end

    end

end

struct SemForwardDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSemSingle
    observed::O
    imply::I 
    loss::L 
    diff::D
    has_gradient::G
end

function (model::SemForwardDiff)(par, F, G, H)

    if H
        model.loss.H .= ForwardDiff.hessian(x -> objective!(model, x), par)
    end

    if model.has_gradient
        model.imply(par, F, G, false, model)
        model.loss(par, F, G, false, model)
    else

        if G
            model.loss.G .= ForwardDiff.gradient(x -> objective!(model, x), par)
        end

        if F
            model.imply(par, F, false, false, model)
            model.loss(par, F, false, false, model)
        end
        
    end

end

#####################################################################################################
# ensemble models
#####################################################################################################

struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, D, S, FT, GT, HT} <: AbstractSemCollection
    n::N
    sems::T
    weights::V
    diff::D
    start_val::S

    F::FT
    G::GT
    H::HT
end

function SemEnsemble(T::Tuple, semdiff, start_val; weights = nothing, parameter_type = Float64)
    n = size(T, 1)
    sems = T
    nobs_total = sum([model.observed.n_obs for model in T])
    if isnothing(weights)
        weights = [(model.observed.n_obs-1)/nobs_total for model in T]
    end
    n_par = length(start_val)
    return SemEnsemble(
        n,
        sems,
        weights,
        semdiff,
        start_val,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

function (ensemble::SemEnsemble)(par, F, G, H)

    if H ensemble.H .= 0.0 end
    if G ensemble.G .= 0.0 end
    if F ensemble.F .= 0.0 end
            
    for (model, weight) in zip(ensemble.sems, ensemble.weights)
        model(par, F, G, H)
        if H ensemble.H .+= weight*hessian(model) end
        if G ensemble.G .+= weight*gradient(model) end
        if F ensemble.F .+= weight*objective(model) end
    end

end

#####################################################################################################
# helpers
#####################################################################################################

objective(model::AbstractSem) = model.loss.F[1]
gradient(model::AbstractSem) = model.loss.G
hessian(model::AbstractSem) = model.loss.H

objective(model::SemEnsemble) = model.F[1]
gradient(model::SemEnsemble) = model.G
hessian(model::SemEnsemble) = model.H

function objective!(model::AbstractSem, parameters)
    model(parameters, true, false, false)
    return model.loss.F[1]
end

function gradient!(model::AbstractSem, parameters)
    model(parameters, false, true, false)
    return model.loss.G
end

function gradient!(grad, model::AbstractSem, parameters)
    model(parameters, false, true, false)
    copyto!(grad, model.loss.G)
    return model.loss.G
end

function hessian!(model::AbstractSem, parameters)
    model(parameters, false, false, true)
    return model.loss.H
end

function hessian!(hessian, model::AbstractSem, parameters)
    model(parameters, false, false, true)
    copyto!(hessian, model.loss.H)
    return model.loss.H
end

function objective!(model::SemEnsemble, parameters)
    model(parameters, true, false, false)
    return model.F[1]
end

function gradient!(model::SemEnsemble, parameters)
    model(parameters, false, true, false)
    return model.G
end

function gradient!(grad, model::SemEnsemble, parameters)
    model(parameters, false, true, false)
    copyto!(grad, model.G)
    return model.G
end

function hessian!(model::SemEnsemble, parameters)
    model(parameters, false, false, true)
    return model.H
end

function hessian!(hessian, model::SemEnsemble, parameters)
    model(parameters, false, false, true)
    copyto!(hessian, model.H)
    return model.H
end

#objective(model::AbstractSem, parameters) = objective!(model, parameters)
#gradient(model::AbstractSem, parameters) = gradient!(model, parameters)
#hessian(model::AbstractSem, parameters) = hessian!(model, parameters)

#objective(model::SemEnsemble, parameters) = objective!(model, parameters)
#gradient(model::SemEnsemble, parameters) = gradient!(model, parameters)
#hessian(model::SemEnsemble, parameters) = hessian!(model, parameters)

function objective_gradient!(model::AbstractSem, parameters)
    model(parameters, true, true, false)
    return model.loss.F[1], copy(model.loss.G)
end

function objective_gradient!(model::SemEnsemble, parameters)
    model(parameters, true, true, false)
    return model.F[1], copy(model.G)
end


function objective_gradient!(grad, model::AbstractSem, parameters)
    model(parameters, true, true, false)
    copyto!(grad, model.loss.G)
    return model.loss.F[1]
end

function objective_gradient!(grad, model::SemEnsemble, parameters)
    model(parameters, true, true, false)
    copyto!(grad, model.G)
    return model.F[1]
end