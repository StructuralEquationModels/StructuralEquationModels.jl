"""
For observed data without missings.

# Constructor

    SemObservedData(;
        data,
        observed_vars = nothing,
        specification = nothing,
        kwargs...)

# Arguments
- `data`: observed data -- *DataFrame* or *Matrix*
- `observed_vars::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)
- `specification`: optional SEM specification ([`SemSpecification`](@ref))

# Extended help
## Interfaces
- `nsamples(::SemObservedData)` -> number of observed data points
- `nobserved_vars(::SemObservedData)` -> number of observed (manifested) variables

- `samples(::SemObservedData)` -> observed data
- `obs_cov(::SemObservedData)` -> observed covariance matrix
- `obs_mean(::SemObservedData)` -> observed mean vector
"""
struct SemObservedData{D <: Union{Nothing, AbstractMatrix}, C, S <: Number} <: SemObserved
    data::D
    observed_vars::Vector{Symbol}
    obs_cov::C
    obs_mean::Vector{S}
    nsamples::Int
end

function SemObservedData(;
    data,
    observed_vars::Union{AbstractVector, Nothing} = nothing,
    specification::Union{SemSpecification, Nothing} = nothing,
    observed_var_prefix::Union{Symbol, AbstractString} = :obs,
    kwargs...,
)

    data, obs_vars, _ =
        prepare_data(data, observed_vars, specification; observed_var_prefix)
    obs_mean, obs_cov = mean_and_cov(data, 1)

    if any(ismissing.(data))
        throw(ArgumentError(
            "Your dataset contains missing values.
            Remove missing values or use full information maximum likelihood (FIML) estimation.
            A FIML model can be constructed with
            Sem(
                ...,
                observed = SemObservedMissing,
                loss = SemFIML,
                meanstructure = true
            )"))
    end

    return SemObservedData(data, obs_vars, Symmetric(obs_cov), vec(obs_mean), size(data, 1))
end

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObservedData) = observed.obs_cov
obs_mean(observed::SemObservedData) = observed.obs_mean
