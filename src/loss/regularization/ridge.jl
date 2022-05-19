# (Ridge) regularization

############################################################################
### Types
############################################################################

struct SemRidge{P, W1, W2, GT, HT} <: SemLossFunction
    α::P
    which::W1
    which_H::W2

    gradient::GT
    hessian::HT
end

############################################################################
### Constructors
############################################################################

function SemRidge(;α_ridge, which_ridge, n_par, parameter_type = Float64, imply = nothing, kwargs...)
    if eltype(which_ridge) <: Symbol
        which_ridge = get_identifier_indices(which_ridge, imply)
    end
    which = [CartesianIndex(x) for x in which_ridge]
    which_H = [CartesianIndex(x, x) for x in which_ridge]
    return SemRidge(
        α_ridge,
        which,
        which_H,

        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### methods
############################################################################

objective!(ridge::SemRidge, par, model) = @views ridge.α*sum(x -> x^2, par[ridge.which])

function gradient!(ridge::SemRidge, par, model)
    @views ridge.gradient[ridge.which] .= 2*ridge.α*par[ridge.which]
    return ridge.gradient
end

function hessian!(ridge::SemRidge, par, model)
    @views @. ridge.hessian[ridge.which_H] += ridge.α*2.0
    return ridge.hessian
end

############################################################################
### Recommended methods
############################################################################

update_observed(loss::SemRidge, observed::SemObs; kwargs...) = loss

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemRidge)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end