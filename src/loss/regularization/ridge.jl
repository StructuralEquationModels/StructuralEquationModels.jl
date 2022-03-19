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

function SemRidge(;α_ridge, which_ridge, n_par, parameter_type = Float64, imply = nothing, kwargs...)
    if which_ridge isa Vector{Symbol}
        which_ridge = get_identifier_indices(which_ridge, imply)
    end
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

    if G
        ridge.G[ridge.which] .= 2*ridge.α*par[ridge.which]
    end

    if H
        @views @. ridge.H[ridge.which_H] += ridge.α*2.0
    end

    if F
        ridge.F[1] = ridge.α*sum(par[ridge.which].^2)
    end
    
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemRidge)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end