############################################################################################
### Types
############################################################################################

# Type to store Expectation Maximization result --------------------------------------------
mutable struct EmMVNModel{A, b, B}
    Σ::A
    μ::b
    fitted::B
end

# FIXME type unstable
obs_mean(em::EmMVNModel) = ifelse(em.fitted, em.μ, nothing)
obs_cov(em::EmMVNModel) = ifelse(em.fitted, em.Σ, nothing)

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

- `samples(::SemObservedMissing)` -> observed data
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
struct SemObservedMissing{
    T <: Real,
    S <: Real,
    E <: EmMVNModel,
} <: SemObserved
    data::Matrix{Union{T, Missing}}
    observed_vars::Vector{Symbol}
    nsamples::Int
    patterns::Vector{SemObservedMissingPattern{T, S}}

    em_model::E
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
                obs_colnames = spec_colnames
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

            obs_colnames = obs_colnames[source_to_dest_perm(obs_colnames, spec_colnames)]
            data = data[:, obs_colnames]
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

    # allocate EM model (but don't fit)
    em_model = EmMVNModel(zeros(nobs_vars, nobs_vars), zeros(nobs_vars), false)

    return SemObservedMissing(data, Symbol.(obs_colnames), nsamples, patterns, em_model)
end

############################################################################################
### Recommended methods
############################################################################################

############################################################################################
### Additional methods
############################################################################################

em_model(observed::SemObservedMissing) = observed.em_model
obs_mean(observed::SemObservedMissing) = obs_mean(em_model(observed))
obs_cov(observed::SemObservedMissing) = obs_cov(em_model(observed))
