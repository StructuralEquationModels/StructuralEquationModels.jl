# constant loss function for comparability with other packages

############################################################################
### Types
############################################################################

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