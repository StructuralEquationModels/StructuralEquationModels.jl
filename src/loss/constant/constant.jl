# constant loss function for comparability with other packages

############################################################################
### Types
############################################################################
"""
    SemConstant(;constant_loss, kwargs...)

Constructor for `SemConstant` objects.
Adds the constant `constant_loss` to the objective.
Can be used for comparability to other packages for example.

# Arguments
- `constant_loss::Number`: constant to add to the objective

# Interfaces
Has analytic gradient! and hessian! methods.
"""
struct SemConstant{C} <: SemLossFunction 
    c::C
end

############################################################################
### Constructors
############################################################################

function SemConstant(;constant_loss, kwargs...)
    return SemConstant(constant_loss)
end

############################################################################
### methods
############################################################################

objective!(constant::SemConstant, par, model) = constant.c
gradient!(constant::SemConstant, par, model) = zero(par)
hessian!(constant::SemConstant, par, model) = zeros(eltype(par), length(par), length(par))

############################################################################
### Recommended methods
############################################################################

update_observed(loss::SemConstant, observed::SemObs; kwargs...) = loss