"""
Type alias for [`SemObservedData`](@ref) that has mean and covariance, but no actual data.

For instances of `SemObservedCovariance` [`samples`](@ref) returns `nothing`.
"""
const SemObservedCovariance{B, C} = SemObservedData{Nothing, B, C}

"""
For observed covariance matrices and means.

# Constructor

    SemObservedCovariance(;
        specification,
        obs_cov,
        obs_colnames = nothing,
        meanstructure = false,
        obs_mean = nothing,
        nsamples = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `obs_cov`: observed covariance matrix
- `obs_colnames::Vector{Symbol}`: column names of the covariance matrix
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_mean`: observed mean vector
- `nsamples::Number`: number of samples (observed data points); necessary for fit statistics

# Extended help
## Interfaces
- `nsamples(::SemObservedCovariance)`: number of samples (observed data points)
- `n_man(::SemObservedCovariance)` -> number of manifest variables

- `obs_cov(::SemObservedCovariance)` -> observed covariance matrix
- `obs_mean(::SemObservedCovariance)` -> observed means

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your covariance matrix is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
"""
function SemObservedCovariance(;
    specification::Union{SemSpecification, Nothing} = nothing,
    obs_cov::AbstractMatrix,
    obs_colnames::Union{AbstractVector{Symbol}, Nothing} = nothing,
    spec_colnames::Union{AbstractVector{Symbol}, Nothing} = nothing,
    obs_mean::Union{AbstractVector, Nothing} = nothing,
    meanstructure::Bool = false,
    nsamples::Integer,
    kwargs...,
)
    if !meanstructure && !isnothing(obs_mean)
        throw(ArgumentError("observed means were passed, but `meanstructure = false`"))
    elseif meanstructure && isnothing(obs_mean)
        throw(ArgumentError("`meanstructure = true`, but no observed means were passed"))
    end

    if isnothing(spec_colnames) && !isnothing(specification)
        spec_colnames = observed_vars(specification)
    end

    if !isnothing(spec_colnames) && isnothing(obs_colnames)
        throw(ArgumentError("no `obs_colnames` were specified"))
    end

    if !isnothing(spec_colnames)
        obs2spec_perm = source_to_dest_perm(obs_colnames, spec_colnames)
        obs_cov = obs_cov[obs2spec_perm, obs2spec_perm]
        isnothing(obs_mean) || (obs_mean = obs_mean[obs2spec_perm])
    end

    return SemObservedData(nothing, obs_cov, obs_mean, size(obs_cov, 1), nsamples)
end
