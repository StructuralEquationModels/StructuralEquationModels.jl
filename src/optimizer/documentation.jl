"""
    sem_fit(model::AbstractSem; start_val = start_val, kwargs...)

Fit the model and return a `SemFit` object.

# Arguments
- `model`: `AbstractSem` to fit
- `start_val`: vector of starting values or function to compute starting values. See also [`start_val`](@ref)
- `kwargs...`: keyword arguments, passed to starting value functions
"""
function sem_fit end