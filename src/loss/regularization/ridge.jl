# (Ridge) regularization

############################################################################################
### Types
############################################################################################
"""
Ridge regularization.

# Constructor

    SemRidge(;α_ridge, which_ridge, n_par, parameter_type = Float64, imply = nothing, kwargs...)

# Arguments
- `α_ridge`: hyperparameter for penalty term
- `which_ridge::Vector`: Vector of parameter labels (Symbols) or indices that indicate which parameters should be regularized.
- `n_par::Int`: number of parameters of the model
- `imply::SemImply`: imply part of the model
- `parameter_type`: type of the parameters

# Examples
```julia
my_ridge = SemRidge(;α_ridge = 0.02, which_ridge = [:λ₁, :λ₂, :ω₂₃], n_par = 30, imply = my_imply)
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
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

function SemRidge(;
    α_ridge,
    which_ridge,
    n_par,
    parameter_type = Float64,
    imply = nothing,
    kwargs...,
)
    if eltype(which_ridge) <: Symbol
        if isnothing(imply)
            throw(
                ArgumentError(
                    "When referring to parameters by label, `imply = ...` has to be specified",
                ),
            )
        else
            which_ridge = get_identifier_indices(which_ridge, imply)
        end
    end
    which = [CartesianIndex(x) for x in which_ridge]
    which_H = [CartesianIndex(x, x) for x in which_ridge]
    return SemRidge(
        α_ridge,
        which,
        which_H,
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par),
    )
end

############################################################################################
### methods
############################################################################################

objective!(ridge::SemRidge, par, model) = @views ridge.α * sum(x -> x^2, par[ridge.which])

function gradient!(ridge::SemRidge, par, model)
    @views ridge.gradient[ridge.which] .= 2 * ridge.α * par[ridge.which]
    return ridge.gradient
end

function hessian!(ridge::SemRidge, par, model)
    @views @. ridge.hessian[ridge.which_H] += ridge.α * 2.0
    return ridge.hessian
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss
