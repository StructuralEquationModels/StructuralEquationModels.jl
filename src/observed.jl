struct SemObsCommon{
        A <: Union{AbstractArray, Nothing},
        B <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        D <: AbstractFloat} <: SemObs
    data::A
    obs_cov::B
    obs_mean::C
    n_man::D
end

function SemObsCommon(;
        data = nothing,
        obs_cov = nothing,
        meanstructure = false)
    # if no cov. matrix was given, compute one
    if isnothing(obs_cov) obs_cov = Statistics.cov(data) end
    n_man = Float64(size(obs_cov, 1))
    # if a meanstructure is needed, compute observed means
    meanstructure ? obs_mean = vcat(Statistics.mean(data, dims = 1)...) :
        obs_mean = nothing
    return SemObsCommon(data, obs_cov, obs_mean, n_man)
end
