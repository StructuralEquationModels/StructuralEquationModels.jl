# (Ridge) regularization

############################################################################
### Types
############################################################################

struct SemRidge{P, W1, W2} <: SemLossFunction
    α::P
    which::W1
    which_H::W2
end

############################################################################
### Constructors
############################################################################

function SemRidge(α, which_vec)
    which = [CartesianIndex(x) for x in which_vec]
    which_H = [CartesianIndex(x, x) for x in which_vec]
    return SemRidge(α, which, which_H)
end

############################################################################
### functors
############################################################################

function (ridge::SemRidge)(par, F, G, H, model, weight = nothing)

    if !isnothing(G)
        grad = 2*ridge.α*par[ridge.which]
        if !isnothing(weight) grad = weight*grad end
        G[ridge.which] .+= grad
    end

    if !isnothing(H)
        @views @. @inbounds H[ridge.which_H] += ridge.α*2.0
    end

    if !isnothing(F)
        F = ridge.α*sum(par[ridge.which].^2)
        if !isnothing(weight) F = weight*F end
        return F
    end
    
end