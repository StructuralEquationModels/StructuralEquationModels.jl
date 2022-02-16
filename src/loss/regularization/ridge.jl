# (Ridge) regularization

############################################################################
### Types
############################################################################

struct SemRidge{P, W1, W2, FT, GT, HT} <: SemLossFunction
    α::P
    which::W1
    which_H::W2

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemRidge(;α_ridge, which_ridge, n_par, parameter_type = Float64, kwargs...)
    which = [CartesianIndex(x) for x in which_ridge]
    which_H = [CartesianIndex(x, x) for x in which_ridge]
    return SemRidge(
        α_ridge,
        which,
        which_H,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (ridge::SemRidge)(par, F, G, H, model)

    if !isnothing(G)
        G = 2*ridge.α*par[ridge.which]
        ridge.G[ridge.which] .= G
    end

    if !isnothing(H)
        @views @. ridge.H[ridge.which_H] += ridge.α*2.0
    end

    if !isnothing(F)
        F = ridge.α*sum(par[ridge.which].^2)
        ridge.F[1] = F
    end
    
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemRidge)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end