"""
Type alias for [SemObservedData](@ref) with no data, but with mean and covariance.
"""
SemObservedCovariance{B, C} = SemObservedData{Nothing, B, C}

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

    return SemObservedData(
        nothing,
        Symmetric(obs_cov),
        obs_mean,
        size(obs_cov, 1),
        nsamples,
    )
end
