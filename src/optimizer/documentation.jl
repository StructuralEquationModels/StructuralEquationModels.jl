"""
    sem_fit(model::AbstractSem; start_val = start_val, kwargs...)

Return the fitted `model`.

# Arguments
- `model`: `AbstractSem` to fit
- `start_val`: a vector or a dictionary of starting parameter values,
               or function to compute them (1)
- `kwargs...`: keyword arguments, passed to starting value functions

(1) available functions are `start_fabin3`, `start_simple` and `start_partable`.
For more information, we refer to the individual documentations and the online documentation on [Starting values](@ref).

# Examples
```julia
sem_fit(
    my_model;
    start_val = start_simple,
    start_covariances_latent = 0.5)
```
"""
function sem_fit end

# dispatch on optimizer
function sem_fit(model::AbstractSem; start_val = nothing, kwargs...)
    start_params = prepare_start_params(start_val, model; kwargs...)
    sem_fit(model.optimizer, model, start_params; kwargs...)
end

# fallback method
sem_fit(optimizer::SemOptimizer, model::AbstractSem, start_params; kwargs...) =
    error("Optimizer $(optimizer) support not implemented.")

function prepare_start_params(start_val, model::AbstractSem; kwargs...)
    if isnothing(start_val)
        # default function for starting parameters
        # FABIN3 for single models, simple algorithm for ensembles
        start_val =
            model isa AbstractSemSingle ? start_fabin3(model; kwargs...) :
            start_simple(model; kwargs...)
    end
    if start_val isa AbstractVector
        (length(start_val) == nparams(model)) || throw(
            DimensionMismatch(
                "The length of `start_val` vector ($(length(start_val))) does not match the number of model parameters ($(nparams(model))).",
            ),
        )
    elseif start_val isa AbstractDict
        start_val = [start_val[param] for param in params(model)] # convert to a vector
    else # function
        start_val = start_val(model; kwargs...)
    end
    @assert start_val isa AbstractVector
    @assert length(start_val) == nparams(model)
    return start_val
end

# define a vector of parameter lower bounds: use user-specified vector as is
function lower_bounds(
    bounds::AbstractVector,
    model::AbstractSem;
    default::Number,
    variance_default::Number,
)
    length(bound) == nparams(model) || throw(
        DimensionMismatch(
            "The length of `bounds` vector ($(length(bounds))) does not match the number of model parameters ($(nparams(model))).",
        ),
    )
    return bounds
end

# define a vector of parameter lower bounds given a dictionary and default values
function lower_bounds(
    bounds::Union{AbstractDict, Nothing},
    model::AbstractSem;
    default::Number,
    variance_default::Number,
)
    varparams = Set(variance_params(model.imply.ram_matrices))
    res = [
        begin
            def = in(p, varparams) ? variance_default : default
            isnothing(bounds) ? def : get(bounds, p, def)
        end for p in SEM.params(model)
    ]

    return res
end

# define a vector of parameter upper bounds: use user-specified vector as is
function upper_bounds(bounds::AbstractVector, model::AbstractSem; default::Number)
    length(bound) == nparams(model) || throw(
        DimensionMismatch(
            "The length of `bounds` vector ($(length(bounds))) does not match the number of model parameters ($(nparams(model))).",
        ),
    )
    return bounds
end

# define a vector of parameter lower bounds given a dictionary and default values
function upper_bounds(
    bounds::Union{AbstractDict, Nothing},
    model::AbstractSem;
    default::Number,
)
    res = [
        begin
            def = default
            isnothing(bounds) ? def : get(bounds, p, def)
        end for p in SEM.params(model)
    ]

    return res
end
