"""
Subtype of `SemObs`, can handle observed data without missings or an observed covariance matrix

# Constructor

    SemObsCommon(;
        specification,
        data = nothing,
        obs_cov = nothing,
        data_colnames = nothing,
        meanstructure = false,
        rowwise = false,
        n_obs = nothing,
        ...,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `obs_cov`: observed covariance matrix 
- `data_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame) or covariance matrix
- `meanstructure::Bool`: does the model have a meanstructure?
- `rowwise::Bool`: should the data be stored also as vectors per observation
- `n_obs::Number`: number of observed data points (necessary for fit statistics for covariance based optimization)


# Interfaces
- `n_obs(::SemObsCommon)` -> number of observed data points
- `n_man(::SemObsCommon)` -> number of manifest variables

- `get_data(::SemObsCommon)` -> observed data
- `obs_cov(::SemObsCommon)` -> observed covariance matrix
- `obs_mean(::SemObsCommon)` -> observed means
- `data_rowwise(::SemObsCommon)` -> observed data, stored as vectors per observation

# Extended help
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you make shure your observed data is in the right format.
"""
struct SemObsCommon{
        A <: Union{AbstractArray, Nothing},
        B <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        D <: AbstractFloat,
        O <: Union{AbstractFloat, Nothing},
        R <: Union{AbstractArray, Nothing}} <: SemObs
    data::A
    obs_cov::B
    obs_mean::C
    n_man::D
    n_obs::O
    data_rowwise::R
end

function SemObsCommon(;
        specification,
        data = nothing,
        spec_colnames = nothing,
        obs_cov = nothing,
        data_colnames = nothing,
        meanstructure = false,
        rowwise = false,
        n_obs = nothing,
        kwargs...)

    # sort columns/rows
    if isnothing(spec_colnames) spec_colnames = get_colnames(specification) end

    data, obs_cov = reorder_observed(data, obs_cov, spec_colnames, data_colnames)

    # if no cov. matrix was given, compute one
    if isnothing(obs_cov) obs_cov = Statistics.cov(data) end
    isnothing(data) ? nothing : n_obs = convert(Float64, size(data, 1))
    n_man = Float64(size(obs_cov, 1))
    # if a meanstructure is needed, compute observed means
    meanstructure ? 
        obs_mean = vcat(Statistics.mean(data, dims = 1)...) :
        obs_mean = nothing
    rowwise ? 
        data_rowwise = [data[i, :] for i = 1:convert(Int64, n_obs)] :
        data_rowwise = nothing
    return SemObsCommon(data, obs_cov, obs_mean, n_man, n_obs, data_rowwise)
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObsCommon) = observed.n_obs
n_man(observed::SemObsCommon) = observed.n_man

############################################################################################
### additional methods
############################################################################################

get_data(observed::SemObsCommon) = observed.data
obs_cov(observed::SemObsCommon) = observed.obs_cov
obs_mean(observed::SemObsCommon) = observed.obs_mean
data_rowwise(observed::SemObsCommon) = observed.data_rowwise

############################################################################################
### Additional functions
############################################################################################

# specification colnames
function get_colnames(specification::ParameterTable)
    if !haskey(specification.variables, :sorted_vars) || 
            (length(specification.variables[:sorted_vars]) == 0)
        colnames = specification.variables[:observed_vars]
    else
        is_obs = [var ∈ specification.variables[:observed_vars] for var in specification.variables[:sorted_vars]]
        colnames = specification.variables[:sorted_vars][is_obs]
    end
    return colnames
end

function get_colnames(specification::RAMMatrices)
    if isnothing(specification.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make shure the order of variables in your data is correct!"
        return nothing
    else
        colnames = specification.colnames[specification.F_ind]
        return colnames
    end
end

function get_colnames(specification::Nothing)
    return nothing
end

# reorder data to match spec_colnames ------------------------------------------------------
reorder_observed(data::Nothing, obs_cov, spec_colnames, data_colnames) = 
    reorder_obs_cov(obs_cov, spec_colnames, data_colnames)
reorder_observed(data, obs_cov::Nothing, spec_colnames, data_colnames) = 
    reorder_data(data, spec_colnames, data_colnames)
reorder_observed(data::Nothing, obs_cov, spec_colnames::Nothing, data_colnames) = 
    nothing, obs_cov
reorder_observed(data, obs_cov::Nothing, spec_colnames::Nothing, data_colnames) = 
    data, nothing

# too much or not enough data specified
reorder_observed(data, obs_cov, spec_colnames, data_colnames) = 
    throw(ArgumentError("you specified both an observed dataset and an observed covariance matrix"))
reorder_observed(data::Nothing, obs_cov::Nothing, spec_colnames, data_colnames) = 
    throw(ArgumentError("you specified neither an observed dataset nor an observed covariance matrix"))

# reorder data -----------------------------------------------------------------------------
reorder_data(data::AbstractArray, spec_colnames, data_colnames::Nothing) =
    throw(ArgumentError("please provide column names via the `data_colnames = ...` argument."))

function reorder_data(data::AbstractArray, spec_colnames, data_colnames)
    if !(eltype(data_colnames) <: Symbol)
        throw(ArgumentError("please specify `data_colnames` as a vector of Symbols"))
    end
    if spec_colnames == data_colnames
        return data, nothing
    else
        new_position = [findall(x .== data_colnames)[1] for x in spec_colnames]
        data = Matrix(data[:, new_position])
        return data, nothing
    end
end

reorder_data(data::DataFrame, spec_colnames, data_colnames::Nothing) = Matrix(data[:, spec_colnames]), nothing
reorder_data(data::DataFrame, spec_colnames, data_colnames) = 
    throw(ArgumentError("please provide column names via the `data_colnames = ...` argument."))

# reorder covariance matrices --------------------------------------------------------------
reorder_obs_cov(obs_cov::AbstractArray, spec_colnames, data_colnames::Nothing) =
    throw(ArgumentError("if an observed covariance is given, `data_colnames = ...` has to be specified."))

function reorder_obs_cov(obs_cov::AbstractArray, spec_colnames, data_colnames)
    new_position = [findall(x .== data_colnames)[1] for x in spec_colnames]
    indices = reshape([CartesianIndex(i, j) for j in new_position for i in new_position], size(obs_cov, 1), size(obs_cov, 1))
    obs_cov = obs_cov[indices]
    return nothing, obs_cov
end