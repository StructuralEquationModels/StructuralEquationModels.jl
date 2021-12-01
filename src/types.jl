#####################################################################################################
# Define the basic type system
#####################################################################################################

abstract type AbstractSem end

abstract type SemLossFunction end

struct SemLoss{F <: Tuple}
    functions::F
end

abstract type SemDiff end

abstract type SemObs end

abstract type SemImply end

struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff} <: AbstractSem
    observed::O
    imply::I
    loss::L
    diff::D
end

function (model::Sem)(par, F, G, H, weight = nothing)
    model.imply(par, F, G, H, model)
    model.loss(par, F, G, H, model, weight)
end

function (loss::SemLoss)(par, F, G, H, model, weight)
    for lossfun in loss.functions lossfun(par, F, G, H, model, weight) end
end

#####################################################################################################
# automatic differentiation
#####################################################################################################

struct SemFiniteDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSem
    observed::O
    imply::I
    loss::L
    diff::D
    has_gradient::G
end

function (model::SemFiniteDiff)(par, F, G, H, weight = nothing)
    if !isnothing(G)
        if model.has_gradient
            model.imply(par, nothing, G, nothing, model)
            model.loss(par, nothing, G, nothing, model, weight)
        else
            G .+= FiniteDiff.finite_difference_gradient(x -> model(x, 0.0, weight), par)
        end
    end
    if !isnothing(H)
        H .+= FiniteDiff.finite_difference_hessian(x -> model(x, 0.0, weight), par)
    end
    if !isnothing(F)
        model.imply(par, F, nothing, nothing, model)
        F = model.loss(par, F, nothing, nothing, model, weight)
        return F
    end
end

(model::SemFiniteDiff)(par, F, weight) = (model::SemFiniteDiff)(par, F, nothing, nothing, weight)

struct SemForwardDiff{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff, G} <: AbstractSem
    observed::O
    imply::I 
    loss::L 
    diff::D
    has_gradient::G
end

function (model::SemForwardDiff)(par, F, G, H, weight = nothing)
    if !isnothing(G)
        if !isnothing(has_gradient)
            model.imply(par, nothing, G, nothing, model)
            model.loss(par, nothing, G, nothing, model, weight)
        else
            G .+= ForwardDiff.gradient(x -> model(x, 0.0, weight), par)
        end
    end
    if !isnothing(H)
        H .+= ForwardDiff.hessian(x -> model(x, 0.0, weight), par)
    end
    if !isnothing(F)
        model.imply(par, F, nothing, nothing, model)
        F = model.loss(par, F, nothing, nothing, model, weight)
        return F
    end
end

(model::SemForwardDiff)(par, F, weight) = (model::SemForwardDiff)(par, F, nothing, nothing, weight)

#####################################################################################################
# ensemble models
#####################################################################################################

struct SemEnsemble{N, T <: Tuple, V <: AbstractVector, D, S} <: AbstractSem
    n::N
    sems::T
    weights::V
    diff::D
    start_val::S
end

function SemEnsemble(T::Tuple, diff, start_val)
    n = size(T, 1)
    sems = T
    nobs_total = sum([model.observed.n_obs for model in T])
    weights = [(model.observed.n_obs-1)/nobs_total for model in T]
    return SemEnsemble(n, sems, weights, diff, start_val)
end

function (ensemble::SemEnsemble)(par, F, G, H)
    if !isnothing(F)
        F = zero(eltype(par))
        for i in 1:ensemble.n
            F += ensemble.sems[i](par, F, G, H, ensemble.weights[i])
        end
        return F
    end
    for i in 1:ensemble.n ensemble.sems[i](par, F, G, H, ensemble.weights[i]) end
end