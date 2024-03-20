############################################################################################
### Types
############################################################################################

# Type to store Expectation Maximization result --------------------------------------------
mutable struct EmMVNModel{A, b, B}
    Σ::A
    μ::b
    fitted::B
end

# data associated with the specific pattern of missed variable
struct SemObservedMissingPattern{T,S}
    obs_mask::BitVector     # observed vars mask
    miss_mask::BitVector    # missing vars mask
    nobserved::Int
    nmissed::Int
    rows::Vector{Int}       # rows in original data
    data::Matrix{T}         # non-missing submatrix of data

    obs_mean::Vector{S} # means of observed vars
    obs_cov::Symmetric{S, Matrix{S}}  # covariance of observed vars
end

function SemObservedMissingPattern(
    obs_mask::BitVector,
    rows::AbstractVector{<:Integer},
    data::AbstractMatrix
)
    T = nonmissingtype(eltype(data))

    pat_data = convert(Matrix{T}, view(data, rows, obs_mask))
    if size(pat_data, 1) > 1
        pat_mean, pat_cov = mean_and_cov(pat_data, 1, corrected=false)
        @assert size(pat_cov) == (size(pat_data, 2), size(pat_data, 2))
    else
        pat_mean = reshape(pat_data[1, :], 1, :)
        pat_cov = fill(zero(T), 1, 1)
    end

    miss_mask = .!obs_mask

    return SemObservedMissingPattern{T, eltype(pat_mean)}(
        obs_mask, miss_mask,
        sum(obs_mask), sum(miss_mask),
        rows, pat_data,
        dropdims(pat_mean, dims=1), Symmetric(pat_cov))
end

n_man(pat::SemObservedMissingPattern) = length(pat.obs_mask)
nobserved_vars(pat::SemObservedMissingPattern) = pat.nobserved
nmissed_vars(pat::SemObservedMissingPattern) = pat.nmissed

n_obs(pat::SemObservedMissingPattern) = length(pat.rows)

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
- `n_obs(::SemObservedMissing)` -> number of observed data points
- `n_man(::SemObservedMissing)` -> number of manifest variables

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
struct SemObservedMissing{
        A <: AbstractMatrix,
        P <: SemObservedMissingPattern,
        S <: EmMVNModel
        } <: SemObserved
    data::A
    n_man::Int
    n_obs::Int
    patterns::Vector{P}
    em_model::S
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
        specification::Union{SemSpecification, Nothing},
        data,

        obs_colnames = nothing,
        spec_colnames = nothing,

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

    n_obs, n_man = size(data)

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
    patterns = [SemObservedMissingPattern(pat, rows, data)
                for (pat, rows) in pairs(pattern_to_rows)]
    sort!(patterns, by=nmissed_vars)

    # allocate EM model (but don't fit)
    em_model = EmMVNModel(zeros(n_man, n_man), zeros(n_man), false)

    return SemObservedMissing(data, n_man, n_obs, patterns, em_model)
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObservedMissing) = observed.n_obs
n_man(observed::SemObservedMissing) = observed.n_man

############################################################################################
### Additional methods
############################################################################################
