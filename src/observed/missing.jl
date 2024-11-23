############################################################################################
### Types
############################################################################################

"""
For observed data with missing values.

# Constructor

    SemObservedMissing(;
        specification,
        data,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `nsamples(::SemObservedMissing)` -> number of observed data points
- `nobserved_vars(::SemObservedMissing)` -> number of manifest variables

- `get_data(::SemObservedMissing)` -> observed data
- `em_model(::SemObservedMissing)` -> `EmMVNModel` that contains the covariance matrix and mean vector found via expectation maximization

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
"""
struct SemObservedMissing{A <: AbstractMatrix, P <: SemObservedMissingPattern, T <: Real} <:
       SemObserved
    data::A
    nobs_vars::Int
    nsamples::Int
    patterns::Vector{P}

    obs_cov::Matrix{T}
    obs_mean::Vector{T}
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
    specification::Union{SemSpecification, Nothing},
    data,
    obs_colnames = nothing,
    spec_colnames = nothing,
    kwargs...,
)
    if isnothing(spec_colnames) && !isnothing(specification)
        spec_colnames = observed_vars(specification)
    end

    if !isnothing(spec_colnames)
        if isnothing(obs_colnames)
            try
                data = data[:, spec_colnames]
            catch
                throw(
                    ArgumentError(
                        "Your `data` can not be indexed by symbols. " *
                        "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.",
                    ),
                )
            end
        else
            if data isa DataFrame
                throw(
                    ArgumentError(
                        "You passed your data as a `DataFrame`, but also specified `obs_colnames`. " *
                        "Please make sure the column names of your data frame indicate the correct variables " *
                        "or pass your data in a different format.",
                    ),
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

    nsamples, nobs_vars = size(data)

    # detect all different missing patterns with their row indices
    pattern_to_rows = Dict{BitVector, Vector{Int}}()
    for (i, datarow) in zip(axes(data, 1), eachrow(data))
        pattern = BitVector(.!ismissing.(datarow))
        if sum(pattern) > 0 # skip all-missing rows
            pattern_rows = get!(() -> Vector{Int}(), pattern_to_rows, pattern)
            push!(pattern_rows, i)
        end
    end
    # process each pattern and sort from most to least number of observed vars
    patterns = [
        SemObservedMissingPattern(pat, rows, data) for (pat, rows) in pairs(pattern_to_rows)
    ]
    sort!(patterns, by = nmissed_vars)

    em_cov, em_mean = em_mvn(patterns; kwargs...)

    return SemObservedMissing(data, nobs_vars, nsamples, patterns, em_cov, em_mean)
end

nsamples(observed::SemObservedMissing) = observed.nsamples
nobserved_vars(observed::SemObservedMissing) = observed.nobs_vars

obs_cov(observed::SemObservedMissing) = observed.obs_cov
obs_mean(observed::SemObservedMissing) = observed.obs_mean
