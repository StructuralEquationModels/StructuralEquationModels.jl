############################################################################################
### Types
############################################################################################

"""
    SemObservedMissing{T <: Real, S <: Real} <: SemObserved

[`SemObserved`](@ref) implementation for data with missing values.

# Constructor

    SemObservedMissing(;
        data,
        observed_vars = nothing,
        specification = nothing,
        lazy_cov = true,
        em_kwargs...)

# Arguments
- `data`: observed data
- `observed_vars::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)
- `specification`: optional SEM model specification ([`SemSpecification`](@ref))
- `lazy_cov::Bool`: whether to defer covariance and mean calculation until requested (default: `true`)
- `em_kwargs...`: keyword arguments to pass to the EM algorithm (see [`em_mvn`](@ref))

`SemObservedMissing` could be used in combination with [`SemFIML`](@ref) loss for the
*full information maximum likelihood* (FIML) to fit SEM with missing data.
It could also be used with other loss functions, e.g. [`SemML`](@ref);
in that case the approximated observed covariance and mean would be calculated using
the *EM* algorithm (see [`em_mvn`](@ref)).
"""
struct SemObservedMissing{T <: Real, S <: Real} <: SemObserved
    data::Matrix{Union{T, Missing}}
    observed_vars::Vector{Symbol}
    nsamples::Int
    patterns::Vector{SemObservedMissingPattern{T, S}}

    em_kwargs::AbstractDict
    obs_cov::Ref{Matrix{S}}
    obs_mean::Ref{Vector{S}}
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
    data,
    observed_vars::Union{AbstractVector, Nothing} = nothing,
    specification::Union{SemSpecification, Nothing} = nothing,
    observed_var_prefix::Union{Symbol, AbstractString} = :obs,
    lazy_cov::Bool = true,
    em_kwargs...,
)
    data, obs_vars, _ =
        prepare_data(data, observed_vars, specification; observed_var_prefix)
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

    S = isempty(patterns) ? Float64 : eltype(patterns[1].measured_mean)
    if lazy_cov
        # defer EM covariance and mean calculation until requested
        em_cov_ref = Ref{Matrix{S}}()
        em_mean_ref = Ref{Vector{S}}()
    else
        em_cov, em_mean = em_mvn(patterns; em_kwargs...)
        em_cov_ref, em_mean_ref = Ref(em_cov), Ref(em_mean)
    end

    return SemObservedMissing(
        convert(Matrix{Union{nonmissingtype(eltype(data)), Missing}}, data),
        obs_vars,
        nsamples,
        patterns,
        em_kwargs, # remember EM kwargs for calculate_cov!
        em_cov_ref,
        em_mean_ref,
    )
end

"""
    calculate_cov!(observed::SemObservedMissing; em_kwargs...)

Force calculation of the observed mean and covariance using the EM algorithm.

# Arguments
- `observed`: the observed data with missing values (see [`SemObservedMissing`](@ref))
- `em_kwargs...`: keyword arguments for the EM algorithm (see [`em_mvn`](@ref)),
  the values provided here override the EM arguments passed to the
  [`SemObservedMissing`](@ref) constructor
"""
function calculate_cov!(observed::SemObservedMissing; em_kwargs...)
    em_kwargs = merge(observed.em_kwargs, em_kwargs)
    em_cov, em_mean = em_mvn(observed.patterns; em_kwargs...)
    observed.obs_cov[] = em_cov
    observed.obs_mean[] = em_mean
    return observed
end

function obs_cov(observed::SemObservedMissing{<:Any, S}) where {S}
    isassigned(observed.obs_cov) || calculate_cov!(observed)
    return observed.obs_cov[]::Matrix{S}
end

function obs_mean(observed::SemObservedMissing{<:Any, S}) where {S}
    isassigned(observed.obs_mean) || calculate_cov!(observed)
    return observed.obs_mean[]::Vector{S}
end
