# (Ridge) regularization

############################################################################
### Types
############################################################################

struct SemRidge{P, W1, FT, GT, HT} <: SemLossFunction
    α::P
    which::W1

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemRidge(α, which_vec, n_par; parameter_type = Float64)
    which = [CartesianIndex(x) for x in which_vec]
    return SemRidge(
        α,
        which,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (ridge::SemRidge)(par, F, G, H, model)

    if !isnothing(F)
        F = ridge.α*sum(par[ridge.which].^2)
        ridge.F[1] = F
    end

    if !isnothing(G)
        G = 2*ridge.α*par[ridge.which]
        ridge.G[ridge.which] .= G
    end

    if !isnothing(H)
        stop("hessian for ridge regularization is not implemented")
    end
    
end