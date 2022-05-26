############################################################################################
### Types and Constructor
############################################################################################
"""
Subtype of `SemDiff` that connects to `Optim.jl` as the optimization backend.

# Constructor

    SemDiffOptim(;
        algorithm = LBFGS(), 
        options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), 
        kwargs...)

# Arguments
- `algorithm`: optimization algorithm.
- `options::Optim.Options`: options for the optimization algorithm

# Usage
All algorithms and options from the Optim.jl library are available, for more information see 
the Optim.jl online documentation.

# Interfaces
- `algorithm(diff::SemDiffOptim)`
- `options(diff::SemDiffOptim)`

# Examples
"""
mutable struct SemDiffOptim{A, B} <: SemDiff
    algorithm::A
    options::B
end

SemDiffOptim(;
    algorithm = LBFGS(), 
    options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), 
    kwargs...) = 
    SemDiffOptim(algorithm, options)

############################################################################################
### Recommended methods
############################################################################################

update_observed(diff::SemDiffOptim, observed::SemObs; kwargs...) = diff

############################################################################################
### additional methods
############################################################################################

algorithm(diff::SemDiffOptim) = diff.algorithm
options(diff::SemDiffOptim) = diff.options