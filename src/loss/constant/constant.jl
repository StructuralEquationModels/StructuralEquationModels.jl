# constant loss function for comparability with other packages

############################################################################################
### Types
############################################################################################
"""
Constant loss term. Can be used for comparability to other packages.

# Constructor

    SemConstant(;constant_loss, kwargs...)

# Arguments
- `constant_loss::Number`: constant to add to the objective

# Examples
```julia
    my_constant = SemConstant(constant_loss = 42.0)
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemConstant{C} <: SemLossFunction
    c::C
end

############################################################################################
### Constructors
############################################################################################

function SemConstant(; constant_loss, kwargs...)
    return SemConstant(constant_loss)
end

############################################################################################
### methods
############################################################################################

objective!(constant::SemConstant, par, model) = constant.c
gradient!(constant::SemConstant, par, model) = zero(par)
hessian!(constant::SemConstant, par, model) = zeros(eltype(par), length(par), length(par))

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss_function::SemConstant, observed::SemObserved; kwargs...) =
    loss_function
