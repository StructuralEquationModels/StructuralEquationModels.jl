"""
For observed covariance matrices and means.

# Constructor

    SemObservedCovariance(;
        specification,
        obs_cov,
        obs_colnames = nothing,
        meanstructure = false,
        obs_mean = nothing,
        n_obs = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `obs_cov`: observed covariance matrix
- `obs_colnames::Vector{Symbol}`: column names of the covariance matrix
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_mean`: observed mean vector
- `n_obs::Number`: number of observed data points (necessary for fit statistics)

# Extended help
## Interfaces
- `n_obs(::SemObservedCovariance)` -> number of observed data points
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
struct SemObservedCovariance{B, C, D, O} <: SemObserved
    obs_cov::B
    obs_mean::C
    n_man::D
    n_obs::O
end

function SemObservedCovariance(;
        specification::Union{SemSpecification, Nothing} = nothing,
        obs_cov::AbstractMatrix,

        obs_colnames::Union{AbstractVector{Symbol}, Nothing} = nothing,
        spec_colnames::Union{AbstractVector{Symbol}, Nothing} = nothing,

        obs_mean::Union{AbstractVector, Nothing} = nothing,
        meanstructure::Bool = false,

        n_obs::Union{Number, Nothing} = nothing,

        kwargs...)

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
        obs_cov = reorder_obs_cov(obs_cov, spec_colnames, obs_colnames)
        isnothing(obs_mean) || (obs_mean = reorder_obs_mean(obs_mean, spec_colnames, obs_colnames))
    end

    n_man = Float64(size(obs_cov, 1))

    return SemObservedCovariance(Symmetric(obs_cov), obs_mean, n_man, n_obs)
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObservedCovariance) = observed.n_obs
n_man(observed::SemObservedCovariance) = observed.n_man

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObservedCovariance) = observed.obs_cov
obs_mean(observed::SemObservedCovariance) = observed.obs_mean

############################################################################################
### Additional functions
############################################################################################

# reorder covariance matrices --------------------------------------------------------------
function reorder_obs_cov(obs_cov, spec_colnames, obs_colnames)
    if spec_colnames == obs_colnames
        return obs_cov
    else
        new_position = [findfirst(==(x), obs_colnames) for x in spec_colnames]
        obs_cov = obs_cov[new_position, new_position]
        return obs_cov
    end
end

# reorder means ----------------------------------------------------------------------------

function reorder_obs_mean(obs_mean, spec_colnames, obs_colnames)
    if spec_colnames == obs_colnames
        return obs_mean
    else
        new_position = [findfirst(==(x), obs_colnames) for x in spec_colnames]
        obs_mean = obs_mean[new_position]
        return obs_mean
    end
end
