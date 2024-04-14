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
my_optimizer = SemOptimizerOptim()

# hessian based optimization with backtracking linesearch and modified initial step size
using Optim, LineSearches

my_newton_optimizer = SemOptimizerOptim(
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
mutable struct SemOptimizerOptim{A, B} <: SemOptimizer{:Optim}
    algorithm::A
    options::B
end

SemOptimizer{:Optim}(args...; kwargs...) = SemOptimizerOptim(args...; kwargs...)

SemOptimizerOptim(;
    algorithm = LBFGS(),
    options = Optim.Options(; f_tol = 1e-10, x_tol = 1.5e-8),
    kwargs...,
) = SemOptimizerOptim(algorithm, options)

############################################################################################
### Recommended methods
############################################################################################

update_observed(optimizer::SemOptimizerOptim, observed::SemObserved; kwargs...) = optimizer

############################################################################################
### additional methods
############################################################################################

algorithm(optimizer::SemOptimizerOptim) = optimizer.algorithm
options(optimizer::SemOptimizerOptim) = optimizer.options
