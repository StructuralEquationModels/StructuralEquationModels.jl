"""
For observed data without missings.

# Constructor

    SemObservedData(;
        specification,
        data,
        meanstructure = false,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `n_obs(::SemObservedData)` -> number of observed data points
- `n_man(::SemObservedData)` -> number of manifest variables

- `get_data(::SemObservedData)` -> observed data
- `obs_cov(::SemObservedData)` -> observed.obs_cov
- `obs_mean(::SemObservedData)` -> observed.obs_mean

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
- `compute_covariance::Bool ) = true`: should the covariance of `data` be computed and stored?
"""
struct SemObservedData{A, B, C} <: SemObserved
    data::A
    obs_cov::B
    obs_mean::C
    n_man::Int
    n_obs::Int
end

# error checks
function check_arguments_SemObservedData(kwargs...)
    # data is a data frame,

end


function SemObservedData(;
        specification::Union{SemSpecification, Nothing},
        data,

        obs_colnames = nothing,
        spec_colnames = nothing,

        meanstructure = false,
        compute_covariance = true,

        kwargs...)

    if isnothing(spec_colnames) && !isnothing(specification)
        spec_colnames = observed_vars(specification)
    end

    if !isnothing(spec_colnames)
        if isnothing(obs_colnames)
            try
                data = data[:, spec_colnames]
            catch
                throw(ArgumentError(
                    "Your `data` can not be indexed by symbols. "*
                    "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.")
                    )
            end
        else
            if data isa DataFrame
                throw(ArgumentError(
                    "You passed your data as a `DataFrame`, but also specified `obs_colnames`. "*
                    "Please make sure the column names of your data frame indicate the correct variables "*
                    "or pass your data in a different format.")
                    )
            end

            if !(eltype(obs_colnames) <: Symbol)
                throw(ArgumentError("please specify `obs_colnames` as a vector of Symbols"))
            end

            data = data[:, source_to_dest_perm(obs_colnames, spec_colnames)]
        end
    end

    if data isa DataFrame
        data = Matrix(data)
    end

    return SemObservedData(data,
        compute_covariance ? Symmetric(cov(data)) : nothing,
        meanstructure ? vec(Statistics.mean(data, dims = 1)) : nothing,
        size(data, 2), size(data, 1))
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObservedData) = observed.n_obs
n_man(observed::SemObservedData) = observed.n_man

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObservedData) = observed.obs_cov
obs_mean(observed::SemObservedData) = observed.obs_mean

############################################################################################
### Additional functions
############################################################################################

# permutation that subsets and reorders source to matches the destination order ------------
function source_to_dest_perm(src::AbstractVector, dest::AbstractVector)
    if dest == src # exact match
        return eachindex(dest)
    else
        src_inds = Dict(el => i for (i, el) in enumerate(src))
        return [src_inds[el] for el in dest]
    end
end