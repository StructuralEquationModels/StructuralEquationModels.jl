# constant loss function for comparability with other packages

############################################################################################
### Types
############################################################################################
"""
    SemConstant{C <: Number} <: AbstractLoss

Constant loss term. Can be used for comparability to other packages.

# Constructor

    SemConstant(;constant_loss, kwargs...)

# Arguments
- `constant_loss::Number`: constant to add to the objective

# Examples
```julia
    my_constant = SemConstant(42.0)
```

# Interfaces
Analytic gradients and hessians are available.
"""
struct SemConstant{C <: Number} <: AbstractLoss
    hessianeval::ExactHessian
    c::C

    SemConstant(c::Number) = new{typeof(c)}(ExactHessian(), c)
end

SemConstant(; constant_loss::Number, kwargs...) = SemConstant(constant_loss)

objective(loss::SemConstant, par) = convert(eltype(par), loss.c)
gradient(loss::SemConstant, par) = zero(par)
hessian(loss::SemConstant, par) = zeros(eltype(par), length(par), length(par))
