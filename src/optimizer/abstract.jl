engine(::Type{<:SemOptimizer{E}}) where {E} = E
engine(optimizer::SemOptimizer) = engine(typeof(optimizer))

"""
    SemOptimizer(args...; engine::Symbol = :Optim, kwargs...)

Constructs a `SemOptimizer` object that can be passed to `fit`(@ref) for specifying aspects 
of the numerical optimization involved in fitting a SEM.

The keyword `engine` controlls which Julia package is used, with `:Optim` being the default.
The additional arguments `args...` and `kwargs...` are engine-specific and control further
aspects of the optimization process, such as the algorithm, convergence criteria or constraints.

More engines are available if specific packages are loaded, for example [*NLopt.jl*](https://github.com/JuliaOpt/NLopt.jl) 
(also see [Constrained optimization](@ref) in the online documentation) or [*ProximalAlgorithms.jl*](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
(also see [Regularization](@ref) in the online documentation).
The documentation of available engines (with the packages loaded in the current Julia session) 
is shown in the extended help.
"""
SemOptimizer(args...; engine::Symbol = :Optim, kwargs...) =
    SemOptimizer{engine}(args...; kwargs...)

# fallback optimizer constructor when the engine E is not supported
function SemOptimizer(::Val{E}, args...; kwargs...) where {E}
    if typeof(E) !== Symbol
        throw(ArgumentError("engine argument must be a Symbol."))
    elseif E == :NLOpt
        error("$E optimizer requires \"using NLopt\".")
    elseif E == :Proximal
        error("$E optimizer requires \"using ProximalAlgorithms\".")
    else
        error("$E optimizer engine is not supported.")
    end
end

SemOptimizer{E}(args...; kwargs...) where {E} = SemOptimizer(Val(E), args...; kwargs...)

"""
    optimizer_engines()

Returns a vector of optimizer engines supported by the `engine` keyword argument of
the [`SemOptimizer`](@ref) constructor.

The list of engines depends on the Julia packages loaded (with the `using` directive)
into the current session.
"""
optimizer_engines() = Symbol[engine(opt_type) for opt_type in subtypes(SemOptimizer)]

"""
    fit([optim::SemOptimizer], model::AbstractSem;
            [engine::Symbol], start_val = start_val, kwargs...)

Return the fitted `model`.

# Arguments
- `optim`: [`SemOptimizer`](@ref) to use for fitting.
           If omitted, a new optimizer is constructed as `SemOptimizer(; engine, kwargs...)`.
- `model`: `AbstractSem` to fit
- `engine`: the optimization engine to use, default is `:Optim`
- `start_val`: a vector or a dictionary of starting parameter values,
               or function to compute them (1)
- `kwargs...`: keyword arguments, passed to optimization engine constructor and
               `start_val` function

(1) available functions are `start_fabin3`, `start_simple` and `start_partable`.
For more information, we refer to the individual documentations and
the online documentation on [Starting values](@ref).

# Examples
```julia
fit(my_model;
    start_val = start_simple,
    start_covariances_latent = 0.5)
```

```julia
using Optim

fit(my_model;
    algorithm = BFGS())
```
"""
function fit(optim::SemOptimizer, model::AbstractSem; start_val = nothing, kwargs...)
    start_params = prepare_start_params(start_val, model; kwargs...)
    @assert start_params isa AbstractVector
    @assert length(start_params) == nparams(model)

    fit(optim, model, start_params; kwargs...)
end

fit(model::AbstractSem; engine::Symbol = :Optim, start_val = nothing, kwargs...) =
    fit(SemOptimizer(; engine, kwargs...), model; start_val, kwargs...)

# fallback method
fit(optim::SemOptimizer, model::AbstractSem, start_params; kwargs...) =
    error("Optimizer $(optim) support not implemented.")

# FABIN3 is the default method for single models
prepare_start_params(start_val::Nothing, model::AbstractSemSingle; kwargs...) =
    start_fabin3(model; kwargs...)

# simple algorithm is the default method for ensembles
prepare_start_params(start_val::Nothing, model::AbstractSem; kwargs...) =
    start_simple(model; kwargs...)

# first argument is a function
prepare_start_params(start_val, model::AbstractSem; kwargs...) = start_val(model; kwargs...)

function prepare_start_params(start_val::AbstractVector, model::AbstractSem; kwargs...)
    (length(start_val) == nparams(model)) || throw(
        DimensionMismatch(
            "The length of `start_val` vector ($(length(start_val))) does not match the number of model parameters ($(nparams(model))).",
        ),
    )
    return start_val
end

function prepare_start_params(start_val::AbstractDict, model::AbstractSem; kwargs...)
    return [start_val[param] for param in params(model)] # convert to a vector
end

# get from the ParameterTable (potentially from a different model with match param names)
# TODO: define kwargs that instruct to get values from "estimate" and "fixed"
function prepare_start_params(start_val::ParameterTable, model::AbstractSem; kwargs...)
    res = zeros(eltype(start_val.columns[:start]), nparams(model))
    param_indices = Dict(param => i for (i, param) in enumerate(params(model)))

    for (param, startval) in zip(start_val.columns[:param], start_val.columns[:start])
        (param == :const) && continue
        par_ind = get(param_indices, param, nothing)
        if !isnothing(par_ind)
            isfinite(startval) && (res[par_ind] = startval)
        else
            throw(
                ErrorException(
                    "Model parameter $(param) not found in the parameter table.",
                ),
            )
        end
    end
    return res
end

# prepare a vector of model parameter bounds (BOUND=:lower or BOUND=:lower):
# use the user-specified "bounds" vector "as is"
function prepare_param_bounds(
    ::Val{BOUND},
    bounds::AbstractVector{<:Number},
    model::AbstractSem;
    default::Number,            # unused for vector bounds
    variance_default::Number,   # unused for vector bounds
) where {BOUND}
    length(bounds) == nparams(model) || throw(
        DimensionMismatch(
            "The length of `bounds` vector ($(length(bounds))) does not match the number of model parameters ($(nparams(model))).",
        ),
    )
    return bounds
end

# prepare a vector of model parameter bounds
# given the "bounds" dictionary and default values
function prepare_param_bounds(
    ::Val{BOUND},
    bounds::Union{AbstractDict, Nothing},
    model::AbstractSem;
    default::Number,
    variance_default::Number,
) where {BOUND}
    varparams = Set(variance_params(model.implied.ram_matrices))
    res = [
        begin
            def = in(p, varparams) ? variance_default : default
            isnothing(bounds) ? def : get(bounds, p, def)
        end for p in SEM.params(model)
    ]

    return res
end
