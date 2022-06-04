############################################################################################
### Types and Constructor
############################################################################################
"""
Connects to `Optim.jl` as the optimization backend.

# Constructor

    SemOptimizerOptim(;
        algorithm = LBFGS(), 
        options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), 
        kwargs...)

# Arguments
- `algorithm`: optimization algorithm.
- `options::Optim.Options`: options for the optimization algorithm

# Usage
All algorithms and options from the Optim.jl library are available, for more information see 
the Optim.jl online documentation.

# Examples
```julia
my_diff = SemOptimizerOptim()

# hessian based optimization with backtracking linesearch and modified initial step size
using Optim, LineSearches

my_newton_diff = SemOptimizerOptim(
    algorithm = Newton(
        ;linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang()
    )
)
```

# Extended help

## Interfaces
- `algorithm(::SemOptimizerOptim)`
- `options(::SemOptimizerOptim)`

## Implementation

Subtype of `SemOptimizer`.
"""
mutable struct SemOptimizerOptim{A, B} <: SemOptimizer
    algorithm::A
    options::B
end

SemOptimizerOptim(;
    algorithm = LBFGS(), 
    options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), 
    kwargs...) = 
    SemOptimizerOptim(algorithm, options)

############################################################################################
### Recommended methods
############################################################################################

update_observed(diff::SemOptimizerOptim, observed::SemObserved; kwargs...) = diff

############################################################################################
### additional methods
############################################################################################

algorithm(diff::SemOptimizerOptim) = diff.algorithm
options(diff::SemOptimizerOptim) = diff.options