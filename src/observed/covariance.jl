"""
Subtype of `SemObs`, can handle observed data without missings or an observed covariance matrix

# Constructor

    SemObsCovariance(;
        specification,
        obs_cov,
        obs_colnames = nothing,
        meanstructure = false,
        obs_mean = nothing,
        spec_colnames = nothing,
        n_obs = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `obs_cov`: observed covariance matrix
- `obs_colnames::Vector{Symbol}`: column names of the covariance matrix
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_mean`: observed mean vector
- `spec_colnames::Vector{Symbol}`: overwrites column names of the specification object
- `n_obs::Number`: number of observed data points (necessary for fit statistics)


# Interfaces
- `n_obs(::SemObsCovariance)` -> number of observed data points
- `n_man(::SemObsCovariance)` -> number of manifest variables

- `obs_cov(::SemObsCovariance)` -> observed covariance matrix
- `obs_mean(::SemObsCovariance)` -> observed means

# Implementation
Subtype of `SemObs`

# Extended help
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are shure your covariance matrix is in the right format.
"""
struct SemObsCovariance{B, C, D, O} <: SemObs
    obs_cov::B
    obs_mean::C
    n_man::D
    n_obs::O
end

function SemObsCovariance(;
        specification,
        obs_cov,

        obs_colnames = nothing,
        spec_colnames = nothing,

        obs_mean = nothing,
        meanstructure = false,

        n_obs = nothing,

        kwargs...)


    if !meanstructure & !isnothing(obs_mean)

        throw(ArgumentError("observed means were passed, but `meanstructure = false`"))

    elseif meanstructure & isnothing(obs_mean)

        throw(ArgumentError("`meanstructure = true`, but no observed means were passed"))

    end

    if isnothing(spec_colnames) spec_colnames = get_colnames(specification) end

    if !isnothing(spec_colnames) & isnothing(obs_colnames)

        throw(ArgumentError("no `obs_colnames` were specified"))

    elseif !isnothing(spec_colnames) & !(eltype(obs_colnames) <: Symbol)

        throw(ArgumentError("please specify `obs_colnames` as a vector of Symbols"))

    end

    n_man = Float64(size(obs_cov, 1))

    if !isnothing(spec_colnames)
        obs_cov = reorder_obs_cov(obs_cov, spec_colnames, obs_colnames)
        obs_mean = reorder_obs_mean(obs_mean, spec_colnames, obs_colnames)
    end

    return SemObsCovariance(obs_cov, obs_mean, n_man, n_obs)
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObsCovariance) = observed.n_obs
n_man(observed::SemObsCovariance) = observed.n_man

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObsCovariance) = observed.obs_cov
obs_mean(observed::SemObsCovariance) = observed.obs_mean

############################################################################################
### Additional functions
############################################################################################

# reorder covariance matrices --------------------------------------------------------------
function reorder_obs_cov(obs_cov, spec_colnames, obs_colnames)
    if spec_colnames == obs_colnames
        return obs_cov
    else
        new_position = [findall(x .== obs_colnames)[1] for x in spec_colnames]
        indices = reshape([CartesianIndex(i, j) for j in new_position for i in new_position], size(obs_cov, 1), size(obs_cov, 1))
        obs_cov = obs_cov[indices]
        return obs_cov
    end
end

# reorder means ----------------------------------------------------------------------------
reorder_obs_mean(obs_mean::Nothing, spec_colnames, obs_colnames) = nothing

function reorder_obs_mean(obs_mean, spec_colnames, obs_colnames)
    if spec_colnames == obs_colnames
        return obs_mean
    else
        new_position = [findall(x .== obs_colnames)[1] for x in spec_colnames]
        obs_mean = obs_mean[new_position]
        return obs_mean
    end
end
